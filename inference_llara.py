

import ast
import random
import pandas as pd
import fire
import torch

import json
from tqdm import tqdm
import os

from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import transformers
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from process import get_prompt


def generate_list_from_csv(data_path, id2title_dict, instuction_str, input_prefix_str):
    def parse_item_ids(item_ids_list):
        titles = [id2title_dict[item_id] for item_id in item_ids_list if item_id in id2title_dict]
        return titles

    df = pd.read_csv(data_path)

    df["item_ids"] = df["item_ids"].apply(ast.literal_eval)
    df["user_id"] = df["user_id"].astype(int)

    json_data = []
    for _, row in df.iterrows():
        item_ids_list = row["item_ids"]
        titles = parse_item_ids(item_ids_list)

        input_titles = titles[:-1]
        output_title = titles[-1]

        input_str = input_prefix_str + ", ".join(f'"{title}" [HistoryEmb]' for title in input_titles)
        output_str = f'"{output_title}"'

        json_entry = {
            "instruction": instuction_str,
            "input": f"{input_str}\n ",
            "output": output_str,
            "history_item_ids": item_ids_list[:-1],
        }
        json_data.append(json_entry)

    return json_data


class MlpProjector(nn.Module):
    def __init__(self, llm_size, rec_size=64):
        super().__init__()
        self.mlp_proj = nn.Sequential(nn.Linear(rec_size, llm_size), nn.GELU(), nn.Linear(llm_size, llm_size))

    def forward(self, x):
        x = self.mlp_proj(x)
        return x


class Prompt_Model:
    def __init__(self, model, rec_model_path, projector_path, tokenizer, accelerator):
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.model = model
        self.his_token_id = tokenizer.encode("[HistoryEmb]")[1]

        self.projector = MlpProjector(self.model.config.hidden_size).to(self.accelerator.device)
        self.load_rec_model(os.path.join(rec_model_path, "item_embeddings.pth"))
        self.load_projector(projector_path)

    def load_projector(self, projector_path):
        device = self.accelerator.device
        if projector_path is not None:
            weight_data = torch.load(projector_path, map_location=device)
            self.projector.load_state_dict(weight_data)
        self.projector.to(device)
        self.projector.eval()
        print("Loading Projector Done")

    def load_rec_model(self, rec_model_path):
        device = self.accelerator.device
        weight_data = torch.load(rec_model_path, map_location=device)#['weight']
        item_num, rec_size = weight_data.shape
        self.rec_model_emb = torch.nn.Embedding(item_num, rec_size, padding_idx=0)
        self.rec_model_emb.weight.data = weight_data.to(device)
        self.rec_model_emb.weight.requires_grad = False
        print("Loading Rec model Done")

    def encode_items(self, seq):
        seq = seq.long()
        item_rec_embs = self.rec_model_emb(seq)
        item_txt_embs = self.projector(item_rec_embs)
        return item_txt_embs

    def wrap_emb(self, inputs):
        input_embeds = self.model.get_input_embeddings()(inputs["input_ids"]).detach()
        #input_embeds.requires_grad = False

        his_item_embeds = self.encode_items(inputs["history_item_ids"])
        len_seq = (inputs["history_item_ids"] != 0).sum(dim=1)

        for i in range(inputs["input_ids"].shape[0]):
            if torch.nonzero(inputs["input_ids"][i] == self.his_token_id).shape[0] > 0:
                idx_tensor = (inputs["input_ids"][i] == self.his_token_id).nonzero().view(-1)

                for idx, item_emb in zip(idx_tensor, his_item_embeds[i, : len_seq[i].item()]):
                    input_embeds[i, idx] = item_emb

        return input_embeds

    def forward(self, inputs, history_item_ids):
        history_item_ids = [torch.tensor(_, device=self.accelerator.device) for _ in history_item_ids]
        history_item_ids = pad_sequence(history_item_ids, padding_value=0, batch_first=True).to(self.accelerator.device)
        inputs_dict = {"input_ids": inputs["input_ids"], "history_item_ids": history_item_ids}
        input_embeds = self.wrap_emb(inputs_dict)

        return input_embeds


def main(
    dataset: str = "kindle",
    base_model: str = "../Llama3.2-3B",
    epoch: int = -1,
    method: str = "llara",
    inference_type: str = 'test',
    sample: int = -1,
    batch_size = 8,
):
    rec_model_path = f"item_emb/{dataset}"
    lora_weights_path = f"./save_lora_model/{dataset}/{method}/checkpoint-{epoch*16}"
    if epoch == -1:
        lora_weights_path = f"./save_lora_model/{dataset}/{method}"

    transformers.set_seed(42)
    accelerator = Accelerator()
    accelerator.print("Dataset: ", dataset)

    data_path = os.path.join("data", dataset)

    if inference_type == 'train':
        test_data_path = os.path.join("./data", dataset, 'train', "train_4096.csv")
        predict_file = ""
        num_return_sequences = 1
    elif inference_type == "generate":
        test_data_path = os.path.join("./data", dataset, 'train', "train_4096.csv")
        predict_file = "predict-train" + ".json"
        num_return_sequences = 4
    elif inference_type == "valid":
        test_data_path = os.path.join("./data", dataset, 'train', "valid_512.csv")
        predict_file = ""
        num_return_sequences = 1
    else:
        test_data_path = os.path.join("./data", dataset, 'test', "test_1000.csv")
        predict_file = "predict" + ".json"
        num_return_sequences = 1

    accelerator.print("test_data_path: ", test_data_path)
    instruction_prompt, history_prompt = get_prompt(dataset)

    id2title_path = os.path.join("data", dataset, "id2name4Rec.json")
    with open(id2title_path, "r") as file:
        data = json.load(file)
    id2title_dict = {int(k): v for k, v in data.items()}

    test_data = generate_list_from_csv(
        data_path=test_data_path,
        id2title_dict=id2title_dict,
        instuction_str=instruction_prompt,
        input_prefix_str=history_prompt,
    )
    if sample != -1:
        test_data = random.sample(test_data, sample) 

    result_json_data = os.path.join(f"data/{dataset}/predict/{method}", predict_file)

    accelerator.wait_for_everyone()

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

    if epoch == 0:
        pass
    else:
        model = PeftModel.from_pretrained(model, lora_weights_path, torch_dtype=torch.bfloat16)
        model.merge_and_unload()
    model.generation_config.cache_implementation = "static"
    model.eval()



    def evaluate(batch, prompt_model: Prompt_Model, num_return_sequences=1, max_new_tokens=128):
        prompt_list = [generate_prompt(instruction, input) for instruction, input in zip(batch[0], batch[1])]
        inputs = tokenizer(prompt_list, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)
        input_emb = prompt_model.forward(inputs, batch[2])

        with torch.no_grad():
            generation_config = GenerationConfig(
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=4,
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )

            fake_input_ids = None

            generation_output = model.generate(
                input_ids=fake_input_ids,
                inputs_embeds=input_emb,
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config,
                prefix_allowed_tokens_fn=None,
            )

            output_seq = generation_output.sequences
            output = tokenizer.batch_decode(output_seq, skip_special_tokens=True)
            output = [_.split("Response:\n")[-1] for _ in output]
            real_outputs = [output[i * num_return_sequences : (i + 1) * num_return_sequences] for i in range(len(output) // num_return_sequences)]

        return real_outputs

    def batch(list, batch_size=batch_size):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i : batch_size * (i + 1)]

    instructions = [_["instruction"] for _ in test_data]
    inputs = [_["input"] for _ in test_data]
    history_item_ids = [_["history_item_ids"] for _ in test_data]
    input_dict = {"instructions": instructions, "inputs": inputs, "history_item_ids": history_item_ids}

    if epoch == 0:
        prompt_model = Prompt_Model(
        model,
        rec_model_path,
        None,
        tokenizer,
        accelerator,
    )
    else:
        prompt_model = Prompt_Model(
            model,
            rec_model_path,
            os.path.join(lora_weights_path, "projector.pth"),
            tokenizer,
            accelerator,
        )

    with accelerator.split_between_processes(input_dict) as input_temp:
        outputs = []
        #sequences_scores = []

        for batch1 in tqdm(
            zip(batch(input_temp["instructions"]), batch(input_temp["inputs"]), batch(input_temp["history_item_ids"])),
            total=(len(input_temp["instructions"]) + batch_size - 1) // batch_size,
        ):
            output = evaluate(batch1, prompt_model, num_return_sequences=num_return_sequences)
            outputs.extend(output)
            #sequences_scores.extend(sequences_score)

    outputs = gather_object(outputs)

    assert len(outputs) == len(test_data)

    for i, _ in enumerate(test_data):
        test_data[i]["predict"] = outputs[i]
        test_data[i]["input"] = test_data[i]["input"].replace(" [HistoryEmb]","")
        test_data[i].pop("history_item_ids")
    if inference_type == "train" or inference_type == "valid":
        return test_data 
    if accelerator.is_main_process:
        folder_path = os.path.dirname(result_json_data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(result_json_data, "w") as f:
            json.dump(test_data, f, indent=4)


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    fire.Fire(main)
