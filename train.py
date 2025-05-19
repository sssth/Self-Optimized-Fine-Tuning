import os
import math
import random
from typing import List
import fire
import numpy as np
import sys
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from inference import main as inference
from evaluate_batch import main as evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainerCallback
from trainer.soft_trainer import SOFT_Trainer
from trainer.soft_trainer_utils import DataCollatorForSeq2Seq

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed: int):
    """
    Helper function for setting the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    # model/data params
    base_model: str =  "../Llama3.2-3B",
    dataset_name: str = "",
    train_type: str = "",
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
):
    set_seed(seed)
    accelerator = Accelerator()
    only_shuffle = True
    if train_type == "SFT":
        train_data_path = os.path.join("./data", dataset_name, 'train', "train_4096.json")
        method = train_type
    elif train_type == "SOFT-wo SA":
        train_data_path = os.path.join("./data", dataset_name, 'train', "train_4096_ref_SFT.json")
        method = train_type
    elif train_type == "SOFT":
        train_data_path = os.path.join("./data", dataset_name, 'train', "train_4096_ref_SFT.json")
        method = f"SOFT-{alpha}"
    else:
        sys.exit("train_type error")

    val_data_path = os.path.join("./data", dataset_name, 'train', "valid_512.json")
    output_dir = os.path.join("./save_lora_model", dataset_name, f"{method}")
    print(f"output_dir = {output_dir}")
    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
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

    """定义辅助函数 tokenize """

    def tokenize(prompt, add_eos_token=True):
        split_output = True
        if split_output:
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
        else:
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
            ):
                result["input_ids"].append(tokenizer.eos_token_id)
                result["attention_mask"].append(1)
                result["label"] = result["input_ids"].copy()
        return result


    def generate_and_tokenize_prompt(data_point, is_train=True):
        full_prompt = generate_prompt(data_point, ref=0)
        tokenized_full_prompt = tokenize(full_prompt)

        if "SOFT" in method and is_train:
            ref_prompt_1 = generate_prompt(data_point, ref=1)    
            tokenized_ref_prompt_1 = tokenize(ref_prompt_1)
            tokenized_full_prompt[f"ref_1_input_ids"] = tokenized_ref_prompt_1["input_ids"]
            tokenized_full_prompt[f"ref_1_attention_mask"] = tokenized_ref_prompt_1["attention_mask"]
            tokenized_full_prompt[f"ref_1_label"] = tokenized_ref_prompt_1["label"]

        return tokenized_full_prompt


    label_names = []
    if "SOFT" in method:
        label_names += [f"ref_1_input_ids", f"ref_1_attention_mask", f"ref_1_label"]
    label_names = None if len(label_names) == 0 else label_names

    train_data = load_dataset("json", data_files=train_data_path, split="train")
    val_data = load_dataset("json", data_files=val_data_path, split="train")
    if only_shuffle:
        train_data = train_data.shuffle(seed=seed)
    else:
        train_data = (
            train_data.shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data.shuffle(seed=seed)
        )
    
    train_data = train_data.map(lambda x: generate_and_tokenize_prompt(x))
    val_data = val_data.map(lambda x: generate_and_tokenize_prompt(x, is_train=False))


    
    trainer = SOFT_Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            lr_scheduler_type="constant_with_warmup",
            num_train_epochs=int(num_epochs),
            learning_rate=learning_rate,
            bf16=True,
            tf32=True,
            optim="adamw_torch",
            logging_strategy="steps",
            logging_steps=16,
            save_strategy="steps",
            save_steps=(1 / int(num_epochs)),
            save_on_each_node=False,
            log_on_each_node=False,
            ddp_find_unused_parameters=False if (world_size != 1) else None,
            report_to="tensorboard",
            ddp_backend="nccl",
            local_rank=int(os.environ.get("LOCAL_RANK", -1)),
            seed=seed,
            label_names = label_names,
        ),
        callbacks=[MyCallback(train_type,method,alpha,dataset_name,output_dir)],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        train_type = train_type,
        alpha = alpha,

    )
    model.config.use_cache = False  # 在训练时use_cache是useless
    print(f"Strating Traning {method}")
    trainer.train()
    


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
        predict_train_data = inference(method=self.method,epoch=int(state.epoch),dataset=self.dataset_name,inference_type="train",sample=256)
        dist_ave = evaluate(test_data=predict_train_data, dataset=self.dataset_name, inference_type="train")
        self.dist.append(dist_ave)
        return control
    def on_epoch_end(self, args, state, control, model, **kwargs):
        predict_valid_data = inference(method=self.method,epoch=int(state.epoch),dataset=self.dataset_name,inference_type="valid")
        eval_result = evaluate(test_data=predict_valid_data, dataset=self.dataset_name, inference_type="valid")
        report_result = eval_result["HR@10"]
        if len(self.result) > 0 and report_result <= max(self.result):
            self.patient += 1
            print(f"patient = {self.patient}")
        else:
            self.patient = 0
        self.result.append(report_result)
        if report_result > max(self.result[:-1], default=float('-inf')):
            self.best_epoch = int(state.epoch) 
            self.best_model_state = model.state_dict() 
        print(f"best epoch: {self.best_epoch}")

        if self.patient >= 2:
            print("early stop!")
            control.should_training_stop = True
            model.save_pretrained(self.output_dir)
        if self.train_type != "SOFT":
            return control
        predict_train_data = inference(method=self.method,epoch=int(state.epoch),dataset=self.dataset_name,inference_type="train",sample=256)
        dist_ave = evaluate(test_data=predict_train_data, dataset=self.dataset_name, inference_type="train")
        self.dist.append(dist_ave)     
        
        return control




def generate_prompt(data_point, ref=0, input=False):
    if ref == 0:
        input_data = data_point["input"]
        output_data = data_point["output"]
    elif ref:
        input_data = data_point["input"]
        output_data = data_point[f"reference_{str(ref)}"]

    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{input_data}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{input_data}

### Response:
{output_data}"""
    


if __name__ == "__main__":
    fire.Fire(train)
