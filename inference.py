
import random
import fire
import torch

import json
from tqdm import tqdm
import os

from accelerate import Accelerator
from accelerate.utils import gather_object
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig



def main(
    method: str = "",
    epoch: int = -1,
    dataset: str = "",
    base_model: str = "../Llama3.2-3B",
    batch_size: int = 8,
    inference_type: str = "test",
    sample: int = -1,
):  

    lora_weights_path_SFT = f"./save_lora_model/{dataset}/{method}/checkpoint-{epoch*16}"
    if epoch == -1:
        lora_weights_path_SFT = f"./save_lora_model/{dataset}/{method}"

    accelerator = Accelerator()
    random.seed(42)

    if inference_type == 'train':
        test_data_path = os.path.join("./data", dataset, 'train', "train_4096.json")
        predict_file = ""
        num_return_sequences = 1
    elif inference_type == "generate":
        test_data_path = os.path.join("./data", dataset, 'train', "train_4096.json")
        predict_file = "predict-train" + ".json"
        num_return_sequences = 4
    elif inference_type == "valid":
        test_data_path = os.path.join("./data", dataset, 'train', "valid_512.json")
        predict_file = ""
        num_return_sequences = 1
    else:
        test_data_path = os.path.join("./data", dataset, 'test', "test_1000.json")
        predict_file = "predict" + ".json"
        num_return_sequences = 1
    
    result_json_data = os.path.join("./data", dataset, "predict", method, predict_file)

    print(f"test_data_path = {test_data_path}")
    if predict_file:
        print(f"result_json_data = {result_json_data}")

    with open(test_data_path, "r") as f:
        test_data = json.load(f)
    if sample != -1:
        test_data = random.sample(test_data, sample)

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
        model = PeftModel.from_pretrained(model, lora_weights_path_SFT, torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()

    model.eval()


    def evaluate(instructions, inputs=None, num_return_sequences=1, max_new_tokens=64, **kwargs):
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)
        inputs_dict = {key: value for key, value in inputs.items()}

        original_num_elements = inputs_dict["input_ids"].shape[0] * num_return_sequences

        generation_config = GenerationConfig(
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=4,
                num_return_sequences=num_return_sequences,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=False,
                **kwargs,
            )
        with torch.no_grad():
            generation_config = generation_config
            generation_output = model.generate(**inputs, generation_config=generation_config)
            output_seq = generation_output.sequences
            output = tokenizer.batch_decode(output_seq, skip_special_tokens=True)
            output = [_.split("Response:\n")[-1] for _ in output]


            real_outputs = output[:original_num_elements]
            real_outputs = [
                real_outputs[i * num_return_sequences : (i + 1) * num_return_sequences] for i in range(len(real_outputs) // num_return_sequences)
            ]
        return real_outputs


    def batch(list, batch_size=batch_size):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i : batch_size * (i + 1)]

    outputs = []
    instructions = [_["instruction"] for _ in test_data]
    inputs = [_["input"] for _ in test_data]
    input_dict = {"instructions": instructions, "inputs": inputs}

    with accelerator.split_between_processes(input_dict) as input_temp:
        outputs = []

        for batch1 in tqdm(
            zip(batch(input_temp["instructions"]), batch(input_temp["inputs"])),
            total=(len(input_temp["instructions"]) + batch_size - 1) // batch_size,
        ):
            instructions, inputs = batch1
            output = evaluate(instructions, inputs, num_return_sequences=num_return_sequences)
            outputs.extend(output)

    outputs = gather_object(outputs)

    for i, _ in tqdm(enumerate(test_data)):
        test_data[i]["predict"] = outputs[i]
    
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
