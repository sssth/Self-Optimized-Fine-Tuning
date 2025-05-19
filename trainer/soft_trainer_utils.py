
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from torch.nn.utils.rnn import pad_sequence
import numpy as np

@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if "history_item_ids" in features[0].keys():
            history_item_ids = [feature["history_item_ids"] for feature in features]
        else:
            history_item_ids = None
        if "label" in features[0].keys():
            labels_name = ["label"]
        else:
            labels_name = ["labels"]
        
        if return_tensors is None:
            return_tensors = self.return_tensors

        ref_num = len(set([int(k.split("_")[1]) for k in features[0].keys() if k.startswith("ref")]))

        groundtruth_features = [{k: v for k, v in feature.items()
            if not k.startswith("ref") and not k.endswith(labels_name[0]) and not k.startswith("history")}
            for feature in features]
        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            groundtruth_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        for i in range(1,ref_num+1):
            ref_features = [{k.replace(f"ref_{i}_",""): v for k, v in feature.items()
            if k.startswith(f"ref_{i}") and not k.endswith(labels_name[0]) }
            for feature in features]
            batch_ref = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                ref_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
            batch_ref[f"ref_{i}_input_ids"] = batch_ref["input_ids"]
            batch_ref[f"ref_{i}_attention_mask"] = batch_ref["attention_mask"]
            batch = batch_ref | batch


        # we have to pad the labels manually as we cannot rely on `tokenizer.pad` and we need them to be of the same length to return tensors
        all_labels = []
        
        labels = [feature[labels_name[0]] for feature in features]
        all_labels.append(labels)
        for i in range(1,ref_num+1):
            all_labels.append([feature[f"ref_{i}_{labels_name[0]}"] for feature in features])
            labels_name.append(f"ref_{i}_labels")

        for i in range(len(all_labels)):
            no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
            if all_labels[i] is not None:
                if no_padding:
                    if isinstance(features[0][labels_name[0]], list):
                        batch[labels_name[i]] = list(all_labels[i])
                    else:
                        batch[labels_name[i]] = [np.concatenate([label, []]) for label in all_labels[i]]
                else:
                    max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                    max_label_length = max(len(l) for l in all_labels[i]) if not max_padding else self.max_length
                    if self.pad_to_multiple_of is not None:
                        max_label_length = (
                            (max_label_length + self.pad_to_multiple_of - 1)
                            // self.pad_to_multiple_of
                            * self.pad_to_multiple_of
                        )

                    padding_side = self.tokenizer.padding_side
                    if isinstance(features[0][labels_name[0]], list):
                        batch[labels_name[i]] = [
                            label + [self.label_pad_token_id] * (max_label_length - len(label))
                            if padding_side == "right"
                            else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                            for label in all_labels[i]
                        ]
                    else:
                        batch[labels_name[i]] = [
                            np.concatenate(
                                [
                                    label,
                                    np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                ]
                            )
                            if padding_side == "right"
                            else np.concatenate(
                                [
                                    np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                    label,
                                ]
                            )
                            for label in all_labels[i]
                        ]

            # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
            if batch.get(labels_name[i], None) is not None:
                if return_tensors == "pt":
                    import torch

                    batch[labels_name[i]] = torch.tensor(batch[labels_name[i]], dtype=torch.int64)
                elif return_tensors == "tf":
                    import tensorflow as tf

                    batch[labels_name[i]] = tf.constant(batch[labels_name[i]], dtype=tf.int64)
                else:
                    batch[labels_name[i]] = np.array(batch[labels_name[i]], dtype=np.int64)
            else:
                batch[labels_name[i]] = None


        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids
        if history_item_ids:
            batch["history_item_ids"] = pad_sequence(
                history_item_ids, batch_first=True, padding_value=0
            )

        return batch


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded