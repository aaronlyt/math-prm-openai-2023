"""
just cls loss, add special tokens make nonsense?, no language loss to supervision

"""
import os
import math
import copy
import re
import torch
import numpy as np
from functools import partial
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from datasets import load_dataset

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'

def nexttok_formatting_prompts(examples, tokenizer, max_len):
    """
    Args:
        examples (_type_): _description_
        tokenizer (_type_): _description_
        max_len (_type_): _description_

    Returns:
        _type_: _description_
    """
    posstep_id = tokenizer.convert_tokens_to_ids("<POS_STEP>")
    negstep_id = tokenizer.convert_tokens_to_ids("<NEG_STEP>")
    neutralstep_id = tokenizer.convert_tokens_to_ids("<NEUTRAL_STEP>")
    endstep_id = tokenizer.convert_tokens_to_ids("<END_STEP>")
    # print("--ids--", posstep_id, negstep_id, neutralstep_id, endstep_id)
    problems = ["### Problem: " + ex + "\n ### Solution: " for ex in examples["problem"]]
    trajectories = []
    for ex in examples["trajectories"]:
        if '# Answer\n\n' in ex[-1]["text"]:
            ex[-1]["text"] = 'The answer is: ' + ex[-1]["text"].split('# Answer\n\n')[-1].strip()
        trajectories.append(ex)
    
    # when prediction, using vocab['<END_STEP>']
    solutions = [u'\n'.join(['<START_STEP>'+ x['text'] + '<END_STEP>' for x in ex]) for ex in trajectories]
    inputs_pair = [problem + solution + '\n' for problem, solution in zip(problems, solutions)]
    inputs_tokens = tokenizer(inputs_pair, padding=False, truncation=True, max_length=max_len)
    
    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    for i in range(len(examples["problem"])):
        prompt_tokens = inputs_tokens["input_ids"][i]
        pad_tokens = [tokenizer.pad_token_id] * (max_len - len(prompt_tokens))
        tokens = (prompt_tokens + pad_tokens)
        mask = [1] * len(prompt_tokens) + [0] * len(pad_tokens)
        labels = [-100] * max_len
        endstep_preindices = [idx for idx, token_id in enumerate(tokens) if token_id == endstep_id]
        for j, label_idx in enumerate(endstep_preindices):
            # causual loss will shift one position
            label_idx -= 1
            if examples['trajectories'][i][j]['rating'] == 1:
                labels[label_idx] = posstep_id
            elif examples['trajectories'][i][j]['rating'] == -1:
                labels[label_idx] = negstep_id
            else:
                labels[label_idx] = posstep_id
            
            mask[label_idx] = 0

        batch_input_ids.append(tokens)
        batch_labels.append(labels)
        batch_attention_mask.append(mask)
    
    outputs = {}
    outputs['input_ids'] = batch_input_ids
    outputs['labels'] = batch_labels
    outputs['attention_mask'] = batch_attention_mask
    
    return outputs


def nexttok_formatting_shepherd(examples, tokenizer, max_len):
    """
    Args:
        examples (_type_): _description_
        tokenizer (_type_): _description_
        max_len (_type_): _description_

    Returns:
        _type_: _description_
    """
    posstep_id = tokenizer.convert_tokens_to_ids("<POS_STEP>")
    negstep_id = tokenizer.convert_tokens_to_ids("<NEG_STEP>")
    endstep_id = tokenizer.convert_tokens_to_ids("<END_STEP>")
    # print("--ids--", posstep_id, negstep_id, neutralstep_id, endstep_id)
    problems = ["### Problem: " + ex + "\n ### Solution: " for ex in examples["problem"]]
    
    # when prediction, using vocab['<END_STEP>']
    solutions = [u'\n'.join(['<START_STEP>'+ re.sub(r'STEP \d+:\s*', '', xstep).strip() + '<END_STEP>' for xstep in ex.split('\n')]) for ex in examples["steps"]]
    inputs_pair = [problem + solution + '\n' for problem, solution in zip(problems, solutions)]
    inputs_tokens = tokenizer(inputs_pair, padding=False, truncation=True, max_length=max_len)
    
    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    for i in range(len(examples["problem"])):
        prompt_tokens = inputs_tokens["input_ids"][i]
        pad_tokens = [tokenizer.pad_token_id] * (max_len - len(prompt_tokens))
        tokens = (prompt_tokens + pad_tokens)
        mask = [1] * len(prompt_tokens) + [0] * len(pad_tokens)
        labels = [-100] * max_len
        endstep_preindices = [idx for idx, token_id in enumerate(tokens) if token_id == endstep_id]
        for j, label_idx in enumerate(endstep_preindices):
            # causual loss will shift one position
            label_idx -= 1
            if examples['hard_ratings'][i][j] == 1:
                labels[label_idx] = posstep_id
            elif examples['hard_ratings'][i][j] == 0:
                labels[label_idx] = negstep_id
            
            mask[label_idx] = 0

        batch_input_ids.append(tokens)
        batch_labels.append(labels)
        batch_attention_mask.append(mask)
    
    outputs = {}
    outputs['input_ids'] = batch_input_ids
    outputs['labels'] = batch_labels
    outputs['attention_mask'] = batch_attention_mask
    
    return outputs


def nexttok_lossfunc(outputs, labels, vocab_size, num_items_in_batch=None):
    """
    refer to trainer get_train_sample num_items_in_batch is important, see transformers/loss/loss_utils.py
    Args:
        outputs (_type_): _description_
        labels (_type_): _description_
        vocab_size (_type_): _description_
        num_items_in_batch (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    reduction = "sum" if num_items_in_batch is not None else "mean"
    reduction = "sum"
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = outputs.logits.float()
    # Flatten the tokens, batch_size * seq_len, vocab_size
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)
    # Enable model parallelism
    labels = labels.to(logits.device)
    loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    
    return loss


def compute_metrics(eval_pred):
    # pass
    print(eval_pred)
    pre, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = predictions if isinstance(predictions, np.ndarray) else predictions.numpy()
    labels = labels if isinstance(labels, np.ndarray) else labels.numpy()
    print('--eval data lenght---', len(pre))
    auc = roc_auc_score(labels, pre)
    ll = log_loss(labels, pre)
    acc = accuracy_score(labels, pre > 0.5)
    result ={
        'auc': auc, 
        'll': ll, 
        'acc': acc, 
    } 
    print(result)
    return result


def preprocess_logits_for_metrics(logits, labels, cand_ids):
    """
    cand_ids: the id of the candidate pos_step and neg_step
    Args:
    """
    step_ids = torch.argwhere(torch.bitwise_or(labels == cand_ids[0], labels == cand_ids[1]))
    pred_logits_cand = logits[step_ids[:, 0], step_ids[:, 1]][:, [cand_ids[1], cand_ids[0]]]
    # get the pos_step probability
    probs = torch.softmax(pred_logits_cand, dim=-1)[:, 1]
    # convert into 0-1 label for evaluation
    gold = torch.where(labels[step_ids[:, 0], step_ids[:, 1]] == cand_ids[1], 0, 1)
    print(probs, gold)
    return probs, gold


from transformers.trainer_pt_utils import get_parameter_names
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
class TrainerWoEmdDecay(Trainer):
    def get_decay_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        ALL_LAYERNORM_LAYERS = [torch.nn.LayerNorm, Qwen2RMSNorm]
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name and \
            'lm_head.weight' not in name and 'model.embed_tokens.weight' not in name]
        print(decay_parameters[:2])
        
        # TODO, no lm head weights?
        return decay_parameters


def nexttok_main(args):
    # model intialization
    tra_phrase = "shepherd"
    num_processes = 8
    data_dir = '../datasets_repo/prm800k/prm800k/data/'
    if tra_phrase == 'small':
        model_path = "/t9k/mnt/workspace/llm_models/qwen2.5_1.5b"
        data_files = os.path.join(data_dir, 'small_data.json')
        learning_rate = 5e-6
        save_directory = f'./tmp/trainer_prm_prmsml{learning_rate}/'
    elif tra_phrase == 'large':
        # model_path = "../scripts/tmp/prm_prmsml5e-06/"
        model_path = "/t9k/mnt/workspace/llm_models/qwen2.5_1.5b"
        data_files = os.path.join(data_dir, 'large_prod_data.json')
        learning_rate = 1e-5 * num_processes
        save_directory = f'./tmp/woembdecay_ep2_prmlgeprod_nttk_{learning_rate}/'
    elif tra_phrase == 'shepherd':
        base_dir = "../datasets_repo/prm800k/"
        data_files = os.path.join(base_dir, 'shepsherd_data/shepherd_annsteps.json')
        model_path = "../scripts/tmp/trainersc_prmlgeprod_nttk_8e-05/checkpoint-2214/"
        learning_rate = 1e-5 * num_processes
        save_directory = f'./tmp/woembdecay_ep5_shepsherd_{learning_rate}/'
    
    # Define the special token and Add the special token to the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if "prm" not in model_path:
        special_tokens_dict = {
            'additional_special_tokens': 
                ['<POS_STEP>', '<NEG_STEP>', '<NEUTRAL_STEP>',
                    '<START_STEP>', '<END_STEP>'
                ]
            }
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"We have added {num_added_toks} tokens, total tokens: {len(tokenizer)}")
    posstep_id = tokenizer.convert_tokens_to_ids("<POS_STEP>")
    negstep_id = tokenizer.convert_tokens_to_ids("<NEG_STEP>")
    
    config = AutoConfig.from_pretrained(model_path)
    config.attention_dropout = 0.1
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True, \
        attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    
    # dataset and tokenization
    train_dset_ = load_dataset("json", data_files=data_files)['train']
    # train_dset_ = train_dset_.map(lambda x: nexttok_formatting_prompts(x, tokenizer, 1024), batched=True, num_proc=12)
    train_dset_ = train_dset_.map(lambda x: nexttok_formatting_shepherd(x, tokenizer, 1024), batched=True, num_proc=12)
    
    # must have, otherwise, the batch will crash
    train_dset_.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    tmp_path = f"../datasets_repo/prm800k/tmp_datas/prm_tmp{tra_phrase}.jsonl"
    train_dset_.select(range(1000)).to_json(tmp_path, orient='records', lines=True)
    
    # train_dset_ = train_dset_.select(range(1000))
    datasets_ = train_dset_.train_test_split(test_size=0.1)
    train_dset, eval_dset = datasets_['train'], datasets_['test']
    
    #SFTTrainer Pre-process the datasets only once per node. The remaining processes will use the cache. with PartialState().local_main_process_first()
    # to test average_tokens_across_devices=True
    trainer = TrainerWoEmdDecay(
        model,
        train_dataset=train_dset,
        eval_dataset=eval_dset,
        args=TrainingArguments(output_dir=save_directory, bf16=True, do_train=True, learning_rate=learning_rate, \
            lr_scheduler_type="cosine_with_min_lr", lr_scheduler_kwargs={"min_lr": 1e-7}, warmup_ratio=0.5, max_grad_norm=1.0, \
            per_device_train_batch_size=4, gradient_accumulation_steps=16, weight_decay=0.1, \
                ddp_find_unused_parameters=False, per_device_eval_batch_size=4, \
                    num_train_epochs=10, save_total_limit=3, eval_strategy="epoch", \
                        save_strategy="epoch", logging_steps=10),
        data_collator=None,
    )
    # trainer.train(resume_from_checkpoint = True)
    trainer.train()

    tokenizer.save_pretrained(save_directory)

if __name__ == '__main__':
    nexttok_main({})