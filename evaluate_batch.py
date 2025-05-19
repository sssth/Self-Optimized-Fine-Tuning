import math
import fire
import torch
import os
import json
from tqdm import tqdm
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer



torch.set_float32_matmul_precision("high")


def main(
    epoch: int = 5,
    dataset: str = "",
    base_model: str = "../Llama3.2-3B",
    batchsize: int = 16,
    method: str = "SFT",
    inference_type: str = 'test',
    test_data: list = [],
):  
    accelerator = Accelerator()
    if inference_type == 'train':
        sequence_num = 1
        predict_path = ""
    elif inference_type == 'generate':
        save_path = os.path.join("./data", dataset, "result", method, f"train.json")
        predict_path = f'./data/{dataset}/predict/{method}/predict-train.json'
        sequence_num = 4
    elif inference_type == 'valid':
        save_path = os.path.join("./data", dataset, "result", method, f"valid.json")
        predict_path = f'./data/{dataset}/predict/{method}/predict-valid.json'
        sequence_num = 1
    elif inference_type == 'test':
        save_path = os.path.join("./data", dataset, "result", method, f"test.json")
        predict_path = f'./data/{dataset}/predict/{method}/predict.json'
        sequence_num = 1

    def batch(list, batch_size):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i : batch_size * (i + 1)]


    def evaluate(rank_tensor, test_data, item_dict, topk_list=[1, 5, 10, 20, 50]):
        NDCG, HR = [], []

        target_items = [item["output"][1:-1].strip(" ") for item in test_data]
        target_item_ids = torch.tensor([item_dict[item] for item in target_items], device="cuda")
        target_item_ranks = rank_tensor[torch.arange(rank_tensor.size(0)), target_item_ids]
        rank_list_tensor = target_item_ranks
        reciprocal_ranks = 1.0 / (rank_list_tensor+1).float()
        for k in topk_list:
            Hit_num = (rank_list_tensor < k).sum().item()
            HR.append(Hit_num / len(test_data))

            mask = rank_list_tensor < k
            NDCG_num = 1 / torch.log(rank_list_tensor[mask] + 2)
            NDCG.append(NDCG_num.sum().item() / len(test_data) / (1.0 / math.log(2)))

        result_dict = dict()
        for i in range(len(topk_list)):
            result_dict["NDCG@" + str(topk_list[i])] = NDCG[i]

        for i in range(len(topk_list)):
            result_dict["HR@" + str(topk_list[i])] = HR[i]


        return result_dict, rank_list_tensor


    def generate_result_file(
        model, tokenizer, predict_path, test_data,  batch_size, item_embedding_table, item_dict, id2name
    ):

        if not test_data:
            f = open(predict_path, "r")
            test_data = json.load(f)
            print(f"len = {len(test_data)}")
            f.close()
        for item in test_data:
            item["predict_truth_id"] = []
            item["predict_truth_item"] = []
            item["predict_truth_dist"] = []
        for num in range(sequence_num):
            text = [_["predict"][num].strip('"').strip(" ") for _ in test_data]

            with torch.no_grad():
                predict_embeddings = []
                for batch_input in tqdm(batch(text, batch_size=batch_size), total=len(text) // batch_size + 1):
                    inputs = tokenizer(batch_input, return_tensors="pt", padding=True).to("cuda")
                    outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                    predict_embeddings.append(hidden_states[-1][:, -1, :].detach())
                predict_embeddings = torch.cat(predict_embeddings, dim=0)  # 5000 x 32000


            dist = torch.cdist(predict_embeddings.cuda(), item_embedding_table.cuda(), p=2) 
            rank = dist.argsort(dim=-1).argsort(dim=-1)

            zero_row_indices = (item_embedding_table==0).all(axis=1).nonzero().squeeze().tolist()
            predict_id = []
            sort_values, sort_indices = rank.sort(dim=1)
            for indice in sort_indices:
                for i in range(len(zero_row_indices)+1):
                    if indice[i].item() not in zero_row_indices:
                        break
                predict_id.append(indice[i].item())
            for i,id in enumerate(predict_id):
                truth_id = item_dict.get(test_data[i]["output"][1:-1], -1)
                test_data[i]["predict_truth_id"].append(id)
                test_data[i]["predict_truth_item"].append(id2name[id])
                test_data[i]["predict_truth_dist"].append(dist[i][truth_id].item())


        if inference_type == 'test':
            result_dict, _ = evaluate(rank, test_data, item_dict)
            print(result_dict)
            folder_path = os.path.dirname(save_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            f = open(save_path, "w")
            json.dump(result_dict, f, indent=4)
            f = open(f'./data/{dataset}/predict/{method}/predict-truth.json', "w")
            json.dump(test_data, f, indent=4)
        elif inference_type == 'valid':
            result_dict, _ = evaluate(rank, test_data, item_dict)
            if accelerator.is_main_process:
                print(f"eval_result: {result_dict}")
            return result_dict
        elif inference_type == 'train':
            dist = [d["predict_truth_dist"][0] for d in test_data]
            return sum(dist) / len(dist)
        elif inference_type == 'generate':
            result_dict, _ = evaluate(rank, test_data, item_dict)
            print(result_dict)
            folder_path = os.path.dirname(save_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            f = open(save_path, "w")
            json.dump(result_dict, f, indent=4)
            f = open(f'./data/{dataset}/predict/{method}/predict-train-truth.json', "w")
            json.dump(test_data, f, indent=4)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    model.eval()
    

    item_embedding_path = os.path.join("./data/", dataset, "item_embedding.pt")
    item_embedding_table = torch.load(item_embedding_path)
    id2name_path = os.path.join("./data/", dataset, "id2name.json")
    with open(id2name_path, "r") as file:
        data = json.load(file)
        
    name2id_dict = {v: int(k) for k, v in data.items()}
    id2name_dict = {int(k): v for k, v in data.items()}

    dist = generate_result_file(
        model,
        tokenizer,
        predict_path,
        test_data,
        batchsize,
        item_embedding_table,
        name2id_dict,
        id2name_dict,
        )
    
    if dist is not None:
        return dist
    


if __name__ == "__main__":
    fire.Fire(main)
