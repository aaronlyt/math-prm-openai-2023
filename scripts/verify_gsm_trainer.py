import os
import sys
import glob
import torch
from functools import partial
from datasets import Dataset, load_dataset
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments
from transformers.data.data_collator import DefaultDataCollator
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tfdataset.gradesm_data import GSMDataset, get_examples
from models.verify_model import Qwen2ForVerifyLM

os.environ['TOKENIZERS_PARALLELISM'] = "false"


def token_process_func(examples, tokenizer, max_len):
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
        # 依赖huggingface causuallm loss自动做label shift
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


def main(args):
    model_path = "/t9k/mnt/workspace/llm_models/qwen2.5_1.5b"
    
    # Define the special token and Add the special token to the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    special_tokens_dict = {'additional_special_tokens': ['<SLU_LABEL>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"We have added {num_added_toks} tokens, total tokens: {len(tokenizer)}")
    
    config = AutoConfig.from_pretrained(model_path)
    config.scale, config.bias = 1.0, 0.0
    config.verify_token_idx = tokenizer.convert_tokens_to_ids('<SLU_LABEL>')
    config.loss_type = 'ForCausalLM'
    model = Qwen2ForVerifyLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, \
        attn_implementation="flash_attention_2")
    model.resize_token_embeddings(len(tokenizer))
    
    # dataset and tokenization
    data_dir = "../datasets_repo/grade-school-math/grade_school_math/verify_data/train_60complete/"
    data_files = glob.glob(data_dir + "*.json")
    train_dset = load_dataset("json", data_files=data_files)['train']
    train_dset = train_dset.map(lambda x: token_process_func(x, tokenizer, 512), batched=True, num_proc=12)
    # must have, otherwise, the batch will crash
    train_dset.set_format(type="torch", columns=["input_ids", "labels","slu_labels", "attention_mask"])

    #SFTTrainer Pre-process the datasets only once per node. The remaining processes will use the cache. with PartialState().local_main_process_first()
    trainer = Trainer(
        model,
        train_dataset=train_dset,
        args=TrainingArguments(output_dir="./verify_trainer", bf16=True, do_train=True, learning_rate=2e-5, \
            per_device_train_batch_size=8, ddp_find_unused_parameters=False, \
                num_train_epochs=2, save_total_limit=2, save_strategy="epoch", logging_steps=200),
        data_collator=DefaultDataCollator()
    )
    trainer.train()    



def formatting_process_func(examples, tokenizer, max_len):
    qns = [ex["answer_sample"].split("\n ### Answer:")[0] + "\n ### Answer:" for ex in examples]
    ans = [ex["answer_sample"].split("\n ### Answer:")[1] for ex in examples]
    solution_labels = [int(ex['batch_sample_label']) for ex in examples]
    
    qns = tokenizer(qns, padding=False, truncation=True, max_length=max_len)
    ans = tokenizer(ans, padding=False, truncation=True, max_length=max_len)
    
    batch_size = len(examples)
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
        slu_labels = [-100] * len(qn_tokens) + slu_tokens + [-100] * len(pad_tokens)
        labels = [-100] * len(qn_tokens) + ans_tokens + [-100] * len(pad_tokens)
        mask = [1] * len(qn_tokens) + [1] * len(ans_tokens) + [0] * len(pad_tokens)
        
        batch_input_ids.append(tokens)
        batch_slu_labels.append(slu_labels)
        batch_labels.append(labels)
        batch_attention_mask.append(mask)
    
    outputs = {}
    outputs['input_ids'] = torch.tensor(batch_input_ids)
    outputs['slu_labels'] = torch.tensor(batch_slu_labels)
    outputs['labels'] = torch.tensor(batch_labels)
    outputs['attention_mask'] = torch.tensor(batch_attention_mask)
    
    
    return outputs

def trlmain(args):
    
    model_path = "/t9k/mnt/workspace/llm_models/qwen2.5_1.5b"
    # Define the special token and Add the special token to the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    special_tokens_dict = {'additional_special_tokens': ['<SLU_LABEL>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"We have added {num_added_toks} tokens, total tokens: {len(tokenizer)}")
    
    config = AutoConfig.from_pretrained(model_path)
    config.scale, config.bias = 1.0, 0.0
    config.verify_token_idx = tokenizer.convert_tokens_to_ids('<SLU_LABEL>')
    config.loss_type = 'ForCausalLM'
    model = Qwen2ForVerifyLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, \
        attn_implementation="flash_attention_2")
    model.resize_token_embeddings(len(tokenizer))
    
    # dataset and tokenization
    data_dir = "../datasets_repo/grade-school-math/grade_school_math/verify_data/train_60complete/"
    data_files = glob.glob(data_dir + "*.json")
    train_dset = load_dataset("json", data_files=data_files)['train']
    
    #SFTTrainer Pre-process the datasets only once per node. The remaining processes will use the cache. with PartialState().local_main_process_first()
    max_len = 512
    trainer = SFTTrainer(
        model,
        train_dataset=train_dset,
        args=SFTConfig(output_dir="./verify_trainer", bf16=True, do_train=True, learning_rate=2e-5, \
            per_device_train_batch_size=8, ddp_find_unused_parameters=False, dataset_num_proc=12, \
                num_train_epochs=2, save_total_limit=2, save_strategy="epoch", logging_steps=200, \
                    dataset_kwargs={"skip_prepare_dataset": True},  remove_unused_columns=False, max_seq_length=max_len),
        data_collator=partial(formatting_process_func, tokenizer=tokenizer, max_len=max_len)
    )
    trainer.train()
    
if __name__ == '__main__':
    trlmain({})