import torch
from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
)

# from utils.qwen import make_context, get_stop_words_ids, decode_tokens, StopWordsLogitsProcessor
import pdb

def build_logits_processor(model, tokenizer, model_name):
    if model_name == 'Llama-2-13B-chat':
        return LogitsProcessorList([])
    elif model_name == 'llava-v1.5-7b':
        return LogitsProcessorList([])
    elif model_name == 'Qwen-14B-Origin':
        stop_words_ids = []
        stop_words_ids.extend(get_stop_words_ids(
            "raw", tokenizer
        ))

        logits_processor = None

        if stop_words_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=stop_words_ids,
                eos_token_id=model.generation_config.eos_token_id,
            )
            if logits_processor is None:
                logits_processor = LogitsProcessorList([stop_words_logits_processor])
            else:
                logits_processor.append(stop_words_logits_processor)
        return logits_processor

def build_stopping_criteriaList(max_length):
    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])
    return stopping_criteria

def build_generation_elements(generation_config, input_ids):
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    eos_token_id = generation_config.eos_token_id
    pad_token_id = generation_config.pad_token_id
    if eos_token_id is None:
        eos_token_id = 151643
    if pad_token_id is None:
        pad_token_id = 151643
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    
    special_tokens = {"eos_token_id": eos_token_id, "pad_token_id": pad_token_id, "eos_token_id_tensor": eos_token_id_tensor}

    return unfinished_sequences, special_tokens

def build_model_kwargs(attention_mask, use_cache=True):
    model_kwargs = {}
    model_kwargs["attention_mask"] = attention_mask
    model_kwargs["use_cache"] = use_cache
    return model_kwargs
