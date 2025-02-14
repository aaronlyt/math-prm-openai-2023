import os
import sys
import glob
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from accelerate import PartialState

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tfdataset.gradesm_data import GSMDataset, get_examples

os.environ['TOKENIZERS_PARALLELISM'] = "false"

def main(args):
    dataset = Dataset.from_list(get_examples("train"))
    model_path = "/t9k/mnt/workspace/llm_models/qwen2.5_1.5b"
    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, \
        attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map={'': device_string})
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['question'])):
            text = f"### Question: {example['question'][i]}\n ### Answer: {example['answer'][i]}"  + "<|endoftext|>"
            output_texts.append(text)
        return output_texts
    
    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    #SFTTrainer Pre-process the datasets only once per node. The remaining processes will use the cache. with PartialState().local_main_process_first()
    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=SFTConfig(output_dir="./tmp/", bf16=True, do_train=True, learning_rate=1e-5, \
            per_device_train_batch_size=8, ddp_find_unused_parameters=False, \
                dataset_num_proc=12, max_seq_length=1024, num_train_epochs=2, \
                    save_total_limit=2, save_strategy="epoch", logging_steps=200),
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )
    trainer.train()
    

def mainsft_prmgen(args):
    data_dir = "../datasets_repo/prm800k/prm800k/sampling_data/generator_data/"
    data_files = glob.glob(data_dir + "*.json")
    dataset = load_dataset("json", data_files=data_files)['train']
    print('--------datatset size-----', dataset.info)
    model_path = "/t9k/mnt/workspace/llm_models/qwen2.5_1.5b"
    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, \
        attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map={'': device_string})
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['problem'])):
            text = f"### Problem: {example['problem'][i]}\n ### Solution: {example['solution_sample'][i].replace('# Answer: ', 'The answer is: ')}"  + "<|endoftext|>"
            output_texts.append(text)
        return output_texts
    
    response_template = " ### Solution:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    
    #SFTTrainer Pre-process the datasets only once per node. The remaining processes will use the cache. with PartialState().local_main_process_first()
    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=SFTConfig(output_dir="./tmp/prm_sftgen_2ep/", bf16=True, do_train=True, learning_rate=5e-6, \
            per_device_train_batch_size=4, gradient_accumulation_steps=8, weight_decay=0.1, ddp_find_unused_parameters=False, \
                dataset_num_proc=12, max_seq_length=1024, num_train_epochs=2, \
                    save_total_limit=1, save_strategy="epoch", logging_steps=200),
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )
    trainer.train()
    
    
if __name__ == '__main__':
    # main({})
    mainsft_prmgen({})