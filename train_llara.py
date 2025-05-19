import ast
import json
import math
import os
from typing import List

import fire
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
import transformers
from accelerate import Accelerator


from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainerCallback

from process import get_prompt
from trainer.soft_trainer_utils import DataCollatorForSeq2Seq
from inference_llara import main as inference_llara
from inference import main as inference
from evaluate_batch import main as evaluate

class MlpProjector(nn.Module):
    def __init__(self, llm_size, rec_size=64):
        super().__init__()
        self.mlp_proj = nn.Sequential(nn.Linear(rec_size, llm_size), nn.GELU(), nn.Linear(llm_size, llm_size))

    def forward(self, x):
        x = self.mlp_proj(x)
        return x


def generate_list_from_csv(train_data_path_csv, train_data_path_json, id2title_dict, instuction_str, input_prefix_str):
    def parse_item_ids(item_ids_list):
        titles = [id2title_dict[item_id] for item_id in item_ids_list if item_id in id2title_dict]
        return titles

    df = pd.read_csv(train_data_path_csv)

    df["item_ids"] = df["item_ids"].apply(ast.literal_eval)
    df["user_id"] = df["user_id"].astype(int)

    f = open(train_data_path_json, "r")
    data = json.load(f)
    json_data = []
    for i, row in df.iterrows():
        item_ids_list = row["item_ids"]
        titles = parse_item_ids(item_ids_list)

        input_titles = titles[:-1]
        output_title = titles[-1]

        input_str = input_prefix_str + ", ".join(f'"{title}" [HistoryEmb]' for title in input_titles)
        if not "mindist" in train_data_path_json:
            output_str = f'"{output_title}"'
        else:
            output_str = data[i]["output"]
        json_entry = {
            "instruction": instuction_str,
            "input": f"{input_str}\n ",
            "output": output_str,
            "history_item_ids": item_ids_list[:-1],
        }
        if "ref" in train_data_path_json:
            json_entry["reference_1"] = data[i]["reference_1"]
        json_data.append(json_entry)

    return json_data


class CustomTrainer(transformers.Trainer):
    def __init__(
        self,
        *args,
        tau=1,
        alpha=1,
        train_type="SFT",
        his_token_id=None,
        rec_model_path=None,
        accelerator=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.alpha = alpha
        self.train_type = train_type
        self.his_token_id = his_token_id
        self.accelerator = accelerator
        self.projector = MlpProjector(self.model.config.hidden_size).cuda()
        self.load_rec_model(os.path.join(rec_model_path, "item_embeddings.pth"))
        
    def create_optimizer(self):
        opt_model = self.model
        decay_parameters = self.get_decay_parameter_names(opt_model)
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for p in self.projector.parameters() if p.requires_grad],
                "weight_decay": self.args.weight_decay,
            }
        ]
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def load_rec_model(self, rec_model_path):
        device = self.accelerator.device
        
        weight_data = torch.load(rec_model_path, map_location=device)#['weight']
        item_num, rec_size = weight_data.shape
        self.rec_model_emb = torch.nn.Embedding(item_num, rec_size, padding_idx=0)
        self.rec_model_emb.weight.data = weight_data.to(device)
        self.rec_model_emb.weight.requires_grad = False
        print("Loading Rec model Done")

    def save_projector(self, save_path):
        model_to_save = self.accelerator.unwrap_model(self.projector)
        self.accelerator.save(model_to_save.state_dict(), save_path)
        
    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call=_internal_call)

        if output_dir is not None:
            projector_save_path = os.path.join(output_dir, "projector.pth")
            self.save_projector(projector_save_path)

    def encode_items(self, seq):
        seq = seq.long()
        item_rec_embs = self.rec_model_emb(seq)
        item_txt_embs = self.projector(item_rec_embs)
        return item_txt_embs

    def wrap_emb(self, inputs):
        input_embeds = self.model.get_input_embeddings()(inputs["input_ids"])
        input_embeds.requires_grad = False

        his_item_embeds = self.encode_items(inputs["history_item_ids"])
        len_seq = (inputs["history_item_ids"] != 0).sum(dim=1)

        for i in range(inputs["input_ids"].shape[0]):
            if torch.nonzero(inputs["input_ids"][i] == self.his_token_id).shape[0] > 0:
                idx_tensor = (inputs["input_ids"][i] == self.his_token_id).nonzero().view(-1)
                #if not idx_tensor.shape[0] == len_seq[i].item():
                #    idx_tensor = idx_tensor[:len_seq[i].item()]
                
                #assert idx_tensor.shape[0] == len_seq[i].item()
                for idx, item_emb in zip(idx_tensor, his_item_embeds[i, : len_seq[i].item()]):
                    input_embeds[i, idx] = item_emb

        return input_embeds

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        truth_inputs = {"input_ids":inputs["input_ids"],
                    "attention_mask":inputs["attention_mask"],
                    "labels":inputs["label"],
                    "history_item_ids":inputs["history_item_ids"]}
        truth_input_embeds = self.wrap_emb(truth_inputs)
        

        SFT_loss = self.compute_loss_from_logits(model, truth_inputs, truth_input_embeds)
        if self.train_type == "SFT":
            loss = SFT_loss
        elif self.train_type == "SOFT-wo SA":
            ref_1_inputs = {"input_ids":inputs["ref_1_input_ids"],
                    "attention_mask":inputs["ref_1_attention_mask"],
                    "labels":inputs["ref_1_labels"],
                    "history_item_ids":inputs["history_item_ids"]}
            ref_1_input_embeds = self.wrap_emb(ref_1_inputs)
            ref_loss = self.compute_loss_from_logits(model, ref_1_inputs, ref_1_input_embeds)
            loss = ref_loss
        elif self.train_type == "SOFT":
            ref_1_inputs = {"input_ids":inputs["ref_1_input_ids"],
                    "attention_mask":inputs["ref_1_attention_mask"],
                    "labels":inputs["ref_1_labels"],
                    "history_item_ids":inputs["history_item_ids"]}
            ref_1_input_embeds = self.wrap_emb(ref_1_inputs)
            ref_loss = self.compute_loss_from_logits(model, ref_1_inputs, ref_1_input_embeds)
            curent_epoch = math.floor(self.state.epoch) + 1
            my_callback = self.callback_handler.callbacks[2]
            dist = my_callback.dist[curent_epoch-1]
            dist_origin = my_callback.dist[0]
            epoch_lambda = min(math.e ** (self.alpha * (dist/dist_origin-1)), 1) 
            loss = (1-epoch_lambda) * SFT_loss + epoch_lambda * ref_loss

        return loss
        

    def compute_loss_from_logits(self, model, inputs, input_embeds):
        labels = inputs["labels"]
        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=inputs["attention_mask"],
            return_dict=True,
            labels=inputs["labels"],
        )
        logits = outputs.logits
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)

        mask = shift_labels != -100
        shift_labels = shift_labels[mask]
        shift_logits = shift_logits[mask]

        pos_logits = torch.exp(shift_logits.gather(1, shift_labels.unsqueeze(1)).squeeze(1) / self.tau)
        pos_loss = -torch.log(pos_logits)

        neg_logits = torch.exp(shift_logits / self.tau)
        neg_loss = torch.log(neg_logits.sum(dim=-1))

        loss = (pos_loss + neg_loss).mean()
        return loss



class Prompt_dataset(Dataset):
    def __init__(self, datalist):
        self.datalist = datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return self.datalist[idx]


def train(
    # model/data params
    base_model: str = "../Llama3.2-3B",  # the only required argument
    dataset_name: str = "kindle",
    train_type: str = "SFT",
    alpha: float = 1,
    sample: int = -1,
    seed: int = 42,
    # training hyperparams
    batch_size: int = 256,
    micro_batch_size: int = 8,
    num_epochs: int = 7,
    learning_rate: float = 1e-4,
    cutoff_len: int = 700,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # 自定义参数
    tau: float = 1,
):
    rec_model_path = f"item_emb/{dataset_name}"
    params = locals()
    transformers.set_seed(seed)
    
    accelerator = Accelerator()

    instruction_prompt, history_prompt = get_prompt(dataset_name)

    id2title_path = os.path.join("data", dataset_name, "id2name4Rec.json")
    with open(id2title_path, "r") as file:
        data = json.load(file)
    id2title_dict = {int(k): v for k, v in data.items()}

    train_data_path_csv = os.path.join("data", dataset_name, "train", f"train_4096.csv")
    if train_type == "SFT":
        train_data_path_json = os.path.join("data", dataset_name, "train", f"train_4096.json")
        method = "llara-SFT"
    elif train_type == "SOFT-wo SA":
        train_data_path_json = os.path.join("data", dataset_name, "train", f"train_4096_ref_llara-SFT.json")
        method = "llara-SOFT-wo SA"
    elif train_type == "SOFT":
        train_data_path_json = os.path.join("data", dataset_name, "train", f"train_4096_ref_llara-SFT.json")
        method = f"llara-SOFT-{alpha}"
    train_data = generate_list_from_csv(
        train_data_path_csv=train_data_path_csv,
        train_data_path_json=train_data_path_json,
        id2title_dict=id2title_dict,
        instuction_str=instruction_prompt,
        input_prefix_str=history_prompt,
    )

    output_dir = os.path.join(
        "save_lora_model",
        dataset_name,
        method
    )
    print(output_dir)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    gradient_accumulation_steps = batch_size // micro_batch_size
    if world_size != 1:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    bnb_config = None
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    #model.print_trainable_parameters()

    his_token_id = tokenizer.encode("[HistoryEmb]")[1]

    def tokenize(prompt, add_eos_token=True):
        prompt_input = prompt.split("### Response:\n")[0] + "### Response:\n"
        result_input = tokenizer(
            prompt_input,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        prompt_output = prompt.split("### Response:\n")[1]
        result_output = tokenizer(
            prompt_output,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        result = {"input_ids": result_input["input_ids"] + result_output["input_ids"][1:],
                "attention_mask": result_input["attention_mask"] + result_output["attention_mask"][1:]}
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["label"] = result["input_ids"].copy()
        result["label"][:len(result_input["input_ids"])] = [-100] * (len(result_input["input_ids"]))
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        tokenized_full_prompt["history_item_ids"] = torch.tensor(data_point["history_item_ids"])
        if "SOFT" in train_type:
            ref_prompt = generate_prompt(data_point, ref=1)
            tokenized_ref_prompt = tokenize(ref_prompt)
            tokenized_full_prompt[f"ref_1_input_ids"] = tokenized_ref_prompt["input_ids"]
            tokenized_full_prompt[f"ref_1_attention_mask"] = tokenized_ref_prompt["attention_mask"]
            tokenized_full_prompt[f"ref_1_label"] = tokenized_ref_prompt["label"]
        return tokenized_full_prompt

    train_data = [generate_and_tokenize_prompt(sample) for sample in tqdm(train_data)]
    train_data = Prompt_dataset(train_data)
    label_names = []
    if "SOFT" in method:
        label_names += [f"ref_1_input_ids", f"ref_1_attention_mask", f"ref_1_label"]
    label_names = None if len(label_names) == 0 else label_names

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        callbacks=[MyCallback(train_type,method,alpha,dataset_name,output_dir)],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        tau=tau,
        alpha=alpha,
        train_type = train_type,
        his_token_id=his_token_id,
        rec_model_path=rec_model_path,
        accelerator=accelerator,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            lr_scheduler_type="constant_with_warmup",
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            tf32=True,
            optim="adamw_torch",
            logging_strategy="steps",
            logging_steps=4,
            save_strategy="steps",
            save_steps=(1 / num_epochs),
            save_on_each_node=False,
            log_on_each_node=False,
            ddp_find_unused_parameters=False if (world_size != 1) else None,
            report_to="tensorboard",
            ddp_backend="nccl",
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            seed=seed,
            remove_unused_columns=False,
            label_names=label_names,
        ),
    )
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(output_dir)
    trainer.save_projector(os.path.join(output_dir, "projector.pth"))

class MyCallback(TrainerCallback):
    def __init__(self, train_type, method, alpha, dataset_name, output_dir):
        self.train_type = train_type
        self.method = method
        self.alpha = alpha
        self.dataset_name = dataset_name
        self.dist = []
        self.patient = 0
        self.result = []
        self.output_dir = output_dir
    def on_train_begin(self, args, state, control, model, **kwargs):
        if self.train_type != "SOFT":
            return control
        predict_train_data = inference_llara(method=self.method,epoch=int(state.epoch),dataset=self.dataset_name,inference_type="train",sample=256)
        dist_ave = evaluate(test_data=predict_train_data, dataset=self.dataset_name, inference_type="train")
        self.dist.append(dist_ave) 
        return control
    def on_epoch_end(self, args, state, control, model, **kwargs):
        
        predict_valid_data = inference_llara(method=self.method,epoch=int(state.epoch),dataset=self.dataset_name,inference_type="valid")
        eval_result = evaluate(test_data=predict_valid_data, dataset=self.dataset_name, inference_type="valid")
        report_result = eval_result["HR@10"]
        if len(self.result) > 0 and report_result <= max(self.result):
            self.patient += 1
            print(f"patient = {self.patient}")
        else:
            self.patient = 0
        self.result.append(report_result)
        if report_result > max(self.result[:-1], default=float('-inf')):  # 如果当前结果是最优的
            self.best_epoch = int(state.epoch)  # 记录最佳的 epoch
            self.best_model_state = model.state_dict()  # 保存模型的参数
            #model.save_pretrained(self.output_dir)
        print(f"best epoch: {self.best_epoch}")

        if self.patient >= 2:
            print("early stop!")
            control.should_training_stop = True
            model.save_pretrained(self.output_dir)
        if self.train_type != "SOFT":
            return control
        predict_train_data = inference_llara(method=self.method,epoch=int(state.epoch),dataset=self.dataset_name,inference_type="train",sample=256)
        dist_ave = evaluate(test_data=predict_train_data, dataset=self.dataset_name, inference_type="train")
        self.dist.append(dist_ave)     
        

        return control

def generate_prompt(data_point, ref=0):
    if ref:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["reference_1"]}"""
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(train)
