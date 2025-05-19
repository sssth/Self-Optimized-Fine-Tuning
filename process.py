
import json
import pandas as pd
import random
import string
import fire
from tqdm import tqdm


prompt = {
    "movie":[
        "The user has watched the following movies before: ",
        "Given a list of movies the user has watched before, please recommend a new movie that the user likes to the user."
    ],
    "kindle":[
        "The user has read the following books before: ",
        "Given a list of books the user has read before, please recommend a new book that the user likes to the user."
    ],
    "game":[
        "The user has played the following video games before: ",
        "Given a list of video games the user has played before, please recommend a new video game that the user likes to the user."
    ],
}


def main(
    # model/data params
    process_type: str = "data_process",
    dataset: str = "",
    method: str = "SFT",
):
    with open(f'./data/{dataset}/id2name4Rec.json', 'r') as file:
        item_dict = json.load(file)

    if process_type == "data_process":
        print("Data Processing")
        construct_instruction(dataset, item_dict, f'./data/{dataset}/train/train.csv', f'./data/{dataset}/train/train_4096.json', sample=4096)
        construct_instruction(dataset, item_dict, f'./data/{dataset}/train/valid.csv', f'./data/{dataset}/train/valid_512.json', sample=512)
        construct_instruction(dataset, item_dict, f'./data/{dataset}/test/test.csv', f'./data/{dataset}/test/test_1000.json', sample=1000)
        sample_data(f'./data/{dataset}/train/train.csv', f'./data/{dataset}/train/train_4096.csv', sample=4096)
        sample_data(f'./data/{dataset}/train/valid.csv', f'./data/{dataset}/train/valid_512.csv', sample=512)
        sample_data(f'./data/{dataset}/test/test.csv', f'./data/{dataset}/test/test_1000.csv', sample=1000)
    if process_type == "mindist":
        json_to_json_mindist(f'./data/{dataset}/train/train_4096.json', 
                            f'./data/{dataset}/predict/{method}/predict-train-truth.json', 
                            f'./data/{dataset}/train/train_4096_ref_{method}.json')






def json_to_json_mindist(input_path_truth, input_path_ref, output_path):
    f = open(input_path_truth, "r")
    data_truth = json.load(f)
    f = open(input_path_ref, "r")
    data_ref = json.load(f)
    new_data = []
    dist_ave = []
    for i,item in enumerate(data_ref):
        dist_ave.append(min(item["predict_truth_dist"]))
        new_item = {}
        new_item["instruction"] = item["instruction"]
        new_item["input"] = item["input"].replace(" [HistoryEmb]","")
        min_index = item["predict_truth_dist"].index(min(item["predict_truth_dist"]))
        new_item["output"] = data_truth[i]["output"]
        min_index = 0
        new_item["reference_1"] = item["predict"][min_index]
        new_data.append(new_item)
    with open(output_path, "w") as f:
        print(output_path)
        json.dump(new_data, f, indent=4)


def construct_instruction(dataset, item_dict, input_path, output_path, sample=False):
    data = pd.read_csv(input_path)
    if sample and len(data) >= sample:
        data = data.sample(n=sample, random_state=42).reset_index(drop=True)
    json_list = []
    for index, row in data.iterrows():
        row['item_ids'] = eval(row['item_ids'])
        #row['item_titles'] = eval(row['item_titles'])
        L = len(row['item_ids'])
        history = prompt[dataset.split("-")[0]][0]
        for i in range(L-1):
            if i == 0:
                history += "\"" + item_dict[str(row['item_ids'][i])] + "\""
            else:
                history += ", \"" + item_dict[str(row['item_ids'][i])] + "\""
        target_name = "\"" + item_dict[str(row['item_ids'][-1])] + "\""
        json_list.append({
            "instruction": prompt[dataset.split("-")[0]][1],
            "input": f"{history}\n ",
            "output": target_name,
        })
    print(len(json_list))    
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)


def sample_data(input_path, output_path, sample=False):
    data = pd.read_csv(input_path)
    if sample and len(data) >= sample:
        data = data.sample(n=sample, random_state=42).reset_index(drop=True)
    data.to_csv(output_path, index=False)


def get_prompt(dataset_name):
    if "movie" in dataset_name:
        instruction_prompt = "Given a list of movies the user has watched before, please recommend a new movie that the user likes to the user."
        history_prompt = "The user has watched the following movies before: "
    elif "kindle" in dataset_name:
        instruction_prompt = "Given a list of books the user has read before, please recommend a new book that the user likes to the user."
        history_prompt = "The user has read the following books before: "
    elif "game" in dataset_name:
        instruction_prompt = "Given a list of video games the user has played before, please recommend a new video game that the user likes to the user."
        history_prompt = "The user has played the following video games before: "

    return instruction_prompt, history_prompt


if __name__ == "__main__":
    fire.Fire(main)
