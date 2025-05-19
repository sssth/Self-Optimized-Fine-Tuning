from collections import defaultdict
import json
import os
import pandas as pd
from tqdm import tqdm
import random
import fire

def get_interaction(dataset, core=5, sample=0):
    df = pd.read_csv(f'{dataset}/interaction.csv', header=None, skiprows=1)
    df = df.astype({0: str, 1: str, 2: float, 3: int})
    df.sort_values(by=3, inplace=True)
    df = df.drop_duplicates()
    interaction_item_dict = defaultdict(list)
    interaction_user_dict = defaultdict(list)
    for row in tqdm(df.itertuples(), total=df.shape[1]):
        interaction_item_dict[row[2]].append(row[1])
        interaction_user_dict[row[1]].append((row[2],row[4]))

    print(f"all_item_length = {len(interaction_item_dict)}")
    print(f"all_user_length = {len(interaction_user_dict)}")

    
    chosen_item_dict = {}
    chosen_user_dict = {}
    chosen_item_set = set(interaction_item_dict.keys())
    chosen_user_set = set(interaction_user_dict.keys())
    if os.path.exists(f'{dataset}/id2name_meta.json'):
        with open(f'{dataset}/id2name_meta.json', "r") as f:
            data = json.load(f)
            chosen_item_set = set(data.keys())
    if sample:
        chosen_item_set = set(random.sample(list(chosen_item_set), sample))
    while(True):
        num_inter = 0
        for k,v in tqdm(interaction_item_dict.items()):
            if k not in chosen_item_set:
                continue
            chosen_v = []
            for i in v:
                if i in chosen_user_set:
                    chosen_v.append(i)
            if len(chosen_v) >= core:
                chosen_item_dict[k] = chosen_v
        for k,v in tqdm(interaction_user_dict.items()):
            if k not in chosen_user_set:
                continue
            chosen_v = []
            for i in v:
                if i[0] in chosen_item_set:
                    chosen_v.append(i)
            if len(chosen_v) >= core:
                num_inter += len(chosen_v)
                chosen_user_dict[k] = chosen_v
        if len(chosen_item_set) == len(chosen_item_dict.keys()) and len(chosen_user_set) == len(chosen_user_dict.keys()):
            break
        chosen_item_set = set(chosen_item_dict.keys())
        chosen_user_set = set(chosen_user_dict.keys())
        interaction_item_dict = chosen_item_dict
        interaction_user_dict = chosen_user_dict
        chosen_item_dict = {}
        chosen_user_dict = {}


    print(f"user_num = {len(chosen_user_set)}")
    print(f"item_num = {len(chosen_item_set)}")
    print(f"{core}core_inter_length = {num_inter}")
    os.makedirs(f"{dataset}/{core}core")
    with open(f"{dataset}/{core}core/interaction.json", "w") as f:
        json.dump(chosen_user_dict, f, indent=4)
    id2name = {v:str(i) for i,v in enumerate(chosen_item_set, start=1)}
    with open(f"{dataset}/{core}core/asin2id.json", "w") as f:
        json.dump(id2name, f, indent=4)
    user_id = {v:str(i) for i,v in enumerate(chosen_user_set, start=1)}
    with open(f"{dataset}/{core}core/user_id.json", "w") as f:
        json.dump(user_id, f, indent=4)


def get_meta_data(dataset):
    id2name = {}
    with open(f"{dataset}/metadata.jsonl", 'r') as f:
        for line in tqdm(f):
            obj = json.loads(line.strip())
            if obj["title"] is not None:
                id2name[obj["parent_asin"]] = obj["title"]
    with open(f"{dataset}/id2name_meta.json", "w") as f:
        json.dump(id2name, f, indent=4)

def get_sequence(dataset, split_num=11, core=5):
    data_list = []
    with open(f"{dataset}/{core}core/interaction.json", "r") as f:
        data = json.load(f)
    with open(f"{dataset}/{core}core/asin2id.json", "r") as f:
        asin2id = json.load(f)
    with open(f"{dataset}/{core}core/user_id.json", "r") as f:
        user_id_dict = json.load(f)
    for k,v in tqdm(data.items()):
        user_id = int(user_id_dict[k])
        if len(v) <= split_num:
            item_ids = [int(asin2id[d[0]]) for d in v]
            timestamp = [d[1] for d in v]
            new_sequence = {'user_id':user_id, 'item_ids':item_ids, "timestamp":timestamp[-1]}
            data_list.append(new_sequence)
        else:
            for i in range(0,len(v)-split_num+1):
                item_ids = [int(asin2id[d[0]]) for d in v][i:i+split_num]
                timestamp = [d[1] for d in v][i:i+split_num]
                new_sequence = {'user_id':user_id, 'item_ids':item_ids, "timestamp":timestamp[-1]}
                data_list.append(new_sequence)
    
    df = pd.DataFrame(data_list)
    df_sorted = df.sort_values('timestamp')
    df_sorted.to_csv(f"{dataset}/{core}core/sequence.csv", index=False)

def get_id2name(dataset, core=5):
    with open(f"{dataset}/{core}core/asin2id.json", "r") as f:
        asin2id = json.load(f)
    with open(f"{dataset}/id2name_meta.json", "r") as f:
        asin2name = json.load(f)
    id2name = {}
    for asin,id in tqdm(asin2id.items()):
        name = asin2name[asin]
        id2name[id] = name.strip()

    new_id2name = {}
    seen = set()
    for id,name in tqdm(id2name.items()):
        if name not in seen:
            new_id2name[id] = name
            seen.add(name)
    with open(f"{dataset}/{core}core/id2name4Rec.json", "w") as f:
        json.dump(id2name, f, indent=4)
    with open(f"{dataset}/{core}core/id2name.json", "w") as f:
        json.dump(new_id2name, f, indent=4)

def split_dataset(dataset, core):
    df = pd.read_csv(f"{dataset}/{core}core/sequence.csv")
    num_df = len(df)
    df_train = df[:int(0.8*num_df)]
    df_valid = df[int(0.8*num_df):int(0.9*num_df)]
    df_test = df[int(0.9*num_df):]
    df_train.to_csv(f"{dataset}/{core}core/train.csv", index=False)
    df_valid.to_csv(f"{dataset}/{core}core/valid.csv", index=False)
    df_test.to_csv(f"{dataset}/{core}core/test.csv", index=False)

    
def random_sample_from_dict(data_dict, num_samples):
    sampled_data = {}
    for key, values in data_dict.items():
        if len(values) >= num_samples:
            sampled_data[key] = random.sample(values, num_samples)
        else:
            sampled_data[key] = values
    return sampled_data    


def main(
    dataset: str = ""
):
    get_meta_data(dataset)
    for core in [10]:
        split_num = 11
        get_interaction(dataset, core)
        get_sequence(dataset, split_num, core=core)
        get_id2name(dataset, core)
        split_dataset(dataset, core)

if __name__ == "__main__":
    fire.Fire(main)