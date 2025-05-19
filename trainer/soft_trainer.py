import math
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from dataclasses import dataclass



class SOFT_Trainer(Trainer):
    def __init__(
        self,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        alpha: float = 1,
        train_type: str = "SFT",
        args: TrainingArguments = None,
        **kwargs, 
    ): 
        self.ref_model = ref_model
        self.alpha = alpha,
        self.train_type = train_type,
        if not isinstance(self.alpha, float):
            self.alpha = self.alpha[0]   
        if not isinstance(self.train_type, str):
            self.train_type = self.train_type[0]
        super().__init__(
            args=args,
            **kwargs, 
        )

    def compute_loss(self, model, inputs, num_items_in_batch):
        if self.train_type == "SFT":
            truth_inputs = {"input_ids":inputs["input_ids"],
                    "attention_mask":inputs["attention_mask"],
                    "labels":inputs["label"]}
            SFT_loss = super().compute_loss(model, truth_inputs, num_items_in_batch)
            return SFT_loss
        else:
            truth_inputs = {"input_ids":inputs["input_ids"],
                    "attention_mask":inputs["attention_mask"],
                    "labels":inputs["label"]}
            
            ref_inputs = {"input_ids":inputs["ref_1_input_ids"],
                    "attention_mask":inputs["ref_1_attention_mask"],
                    "labels":inputs["ref_1_labels"]}
            ref_loss = super().compute_loss(model, ref_inputs, num_items_in_batch)

            if self.train_type == "SOFT-wo SA":
                return ref_loss
            SFT_loss = super().compute_loss(model, truth_inputs, num_items_in_batch)
            if self.train_type == "SOFT":
                curent_epoch = math.floor(self.state.epoch) + 1
                my_callback = self.callback_handler.callbacks[2]
                dist = my_callback.dist[curent_epoch-1]
                dist_origin = my_callback.dist[0]
                
                epoch_lambda = min(math.e ** (self.alpha * (dist/dist_origin-1)), 1) 
                loss = (1-epoch_lambda) * SFT_loss + epoch_lambda * ref_loss  
                return loss


    