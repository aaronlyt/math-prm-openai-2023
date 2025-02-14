import os
import sys
import glob
import collections

import torch
import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from accelerate import PartialState, Accelerator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tfdataset.gradesm_data import get_examples


def formatting_prompts_func(examples, tokenizer, max_len):
    qns = [ex.split("\n ### Answer:")[0] + "\n ### Answer:" for ex in examples["answer_sample"]]
    ans = [ex.split("\n ### Answer:")[1] for ex in examples["answer_sample"]]
    solution_labels = [int(ex) for ex in examples['batch_sample_label']]
    
    qns = tokenizer(qns, padding=False, truncation=True, max_length=max_len)
    ans = tokenizer(ans, padding=False, truncation=True, max_length=max_len)
    
    batch_size = len(examples["answer_sample"])
    batch_input_ids = []
    batch_labels = []
    batch_slu_labels = []
    batch_attention_mask = []
    for i in range(batch_size):
        qn_tokens = qns["input_ids"][i]
        ans_tokens = ans["input_ids"][i]
        slu_tokens = [solution_labels[i]] * len(ans_tokens)
        pad_tokens = [tokenizer.pad_token_id] * (max_len - len(qn_tokens) - len(ans_tokens))
        tokens = (qn_tokens + ans_tokens + pad_tokens)
        # # 依赖huggingface causuallm loss自动做label shift
        slu_labels = [-100] * len(qn_tokens) + slu_tokens + [-100] * len(pad_tokens)
        labels = [-100] * len(qn_tokens) + ans_tokens + [-100] * len(pad_tokens)
        mask = [1] * len(qn_tokens) + [1] * len(ans_tokens) + [0] * len(pad_tokens)
        
        batch_input_ids.append(tokens)
        batch_slu_labels.append(slu_labels)
        batch_labels.append(labels)
        batch_attention_mask.append(mask)
    
    outputs = {}
    outputs['input_ids'] = batch_input_ids
    outputs['slu_labels'] = batch_slu_labels
    outputs['labels'] = batch_labels
    outputs['attention_mask'] = batch_attention_mask
    
    return outputs


def slu_loss_func(hidden_states, batch_slu_labels, slu_label_id, scalar=1.0, bias=0.0):
    # Extract the logits at the 10000th position
    logits_at_10000 = hidden_states[:, :, slu_label_id]
    logits_at_10000 = scalar * logits_at_10000 + bias
    logits_at_10000[batch_slu_labels == -100] = -100
    # not tested, previous just use mse_loss with reduction mean
    sum_error = torch.nn.functional.mse_loss(logits_at_10000, batch_slu_labels, reduction='sum')
    mean_error = sum_error / torch.sum(batch_slu_labels != -100)
    return mean_error


def main(args):
    # model intialization
    model_path = "/t9k/mnt/workspace/llm_models/qwen2.5_1.5b"
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, \
        attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    
    # tokenizer initialization
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    # Define the special token and Add the special token to the tokenizer
    special_tokens_dict = {'additional_special_tokens': ['<SLU_LABEL>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"We have added {num_added_toks} tokens, total tokens: {len(tokenizer)}")
    
    model.resize_token_embeddings(len(tokenizer))
    slu_label_id = tokenizer.convert_tokens_to_ids('<SLU_LABEL>')
    
    # dataset and tokenization
    data_dir = "../datasets_repo/grade-school-math/grade_school_math/verify_data/train_60complete/"
    data_files = glob.glob(data_dir + "*.json")
    train_dset = load_dataset("json", data_files=data_files)['train']
    
    # load 
    accelerator = Accelerator()
    with accelerator.main_process_first():
        train_dset = train_dset.map(lambda x: formatting_prompts_func(x, tokenizer, 512), batched=True, num_proc=12)
        # must have, otherwise, the batch will crash
        train_dset.set_format(type="torch", columns=["input_ids", "labels", "slu_labels", "attention_mask"])
    train_loader = DataLoader(train_dset, batch_size=8, shuffle=True)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    num_epochs = 1
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    train_loader, model, optimizer, lr_scheduler = accelerator.prepare(train_loader, model, optimizer, lr_scheduler)

    # device = torch.device("cuda")
    pbar = tqdm(range(num_epochs * len(train_loader)))
    
    for _ in range(num_epochs):
        for batch in train_loader:
            batch_slu_labels = batch.pop("slu_labels")
            batch_slu_labels = batch_slu_labels.type(torch.bfloat16)
            # batch_slu_labels = batch_slu_labels.to(device)
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # causual loss see: https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L32 ForCausalLMLoss
            lan_loss = outputs[0]
            
            logtis = outputs[1]
            slu_loss = slu_loss_func(logtis, batch_slu_labels, slu_label_id, 1, 0)
            loss = lan_loss + slu_loss
            # print('--loss--', lan_loss, slu_loss)
            # loss.backward()
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            pbar.update(1)
            pbar.set_description(f"lan_loss: {lan_loss.item():.5f}, slu_loss: {slu_loss.item():.5f}")

    accelerator.wait_for_everyone()
    save_directory = './tmp/verify_model/checkpoint-000'
    tokenizer.save_pretrained(save_directory)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        save_directory,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    
if __name__ == '__main__':
    main({})