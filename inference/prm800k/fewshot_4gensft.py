"""
two purpose:
- Generator: few-shot generate solutions to MATH training problems
- Outcome-supervised Reward Models (ORMs) dataset sampling
"""

import os
import sys
import random
import json
from typing import Any, Dict, List
import numpy as np

import torch as th
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams

from grader import grade_answer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# https://docs.vllm.ai/en/stable/getting_started/examples/offline_inference_distributed.html
# Set tensor parallelism per instance.
tensor_parallel_size = 1
# Set number of instances. Each instance will use tensor_parallel_size GPUs.
num_instances = 3

class LLMPredictor:
    def __init__(self):
        # Create an LLM.
        model_path = "/llm_models/qwen2.5_1.5b/"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.llm = LLM(model=model_path, trust_remote_code=True, max_model_len=8192, \
            tensor_parallel_size=tensor_parallel_size, enable_prefix_caching=True, enable_chunked_prefill=False)
        
        few_shots = []
        large_data = json.load(open("../../datasets_repo/prm800k/prm800k/data/large_data.json", "r"))
        for example in large_data:
            if example["trajectories"][-1]["rating"] != 1:
                continue
            if "# Answer" not in example["trajectories"][-1]["text"]:
                continue
            answer_line  = example["trajectories"][-1]["text"]
            answer_line = "\n# Answer: " + answer_line.split("# Answer")[1].strip().replace("\n", "")
            trajectory_fromat = u'\n'.join(["STEP " + str(idx) + ": " + x['text'].replace("\n", " ") \
                for idx, x in enumerate(example["trajectories"][:-1])])
            trajectory_fromat += answer_line
            few_shots.append({
                "problem": example["problem"],
                "solution": trajectory_fromat
            })
        few_shots_ = random.sample(few_shots, 4)
        self.fewshot_str = "\n\n\n".join([f"### Problem: {fs['problem']}\n ### Solution: {fs['solution']}" for fs in few_shots_]).strip(" ")
        
        # Create a class to do batch inference. Create a sampling params object.
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "\n\n### Problem:"]
        stop_token_ids = [151645, 151643]
        self.sampling_params = SamplingParams(temperature=0.6, top_p=1, max_tokens=1024, stop_token_ids=stop_token_ids, stop=stop_words)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        problems_rmnewline = [problem.replace("\n", " ") for problem in batch['problem']]
        batch_qn_prompt = [f"{self.fewshot_str}\n\n\n### Problem: {problem}\n ### Solution:" for problem in problems_rmnewline]
        
        outputs = self.llm.generate(batch_qn_prompt, self.sampling_params)
        batch_samples = {}
        for key in batch:
            batch_samples[key] = []
        batch_samples['solution_sample'] = []
        for idx, output in enumerate(outputs):
            if output.outputs[0].finish_reason != 'stop':
                # print('---finish_reason---', output.outputs[0].finish_reason)
                continue
            generated_text = output.outputs[0].text
            if "# Answer: " not in generated_text:
                continue
            pred_answer = generated_text.split("# Answer: ")[1].strip()
            # print('------pred answer---', pred_answer, batch['answer'][idx])
            if not pred_answer:
                continue
            if not grade_answer(pred_answer, batch['answer'][idx]):
                continue
            batch_samples['problem'].append(batch['problem'][idx])
            batch_samples['solution'].append(batch['solution'][idx])
            batch_samples['answer'].append(batch['answer'][idx])
            batch_samples['subject'].append(batch['subject'][idx])
            batch_samples['level'].append(batch['level'][idx])
            batch_samples['unique_id'].append(batch['unique_id'][idx])
            batch_samples['solution_sample'].append(generated_text)
            
        # print("batch_sample_labels", batch_sample_labels)
        return batch_samples

# For tensor_parallel_size > 1, we need to create placement groups for vLLM
# to use. Every actor has to have its own placement group.
def scheduling_strategy_fn():
    # One bundle per tensor parallel worker
    pg = ray.util.placement_group(
        [{
            "GPU": 1,
            "CPU": 4
        }] * tensor_parallel_size,
        strategy="STRICT_PACK",
    )
    return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
        pg, placement_group_capture_child_tasks=True))


if __name__ == "__main__":
    base_dir = "../../datasets_repo/prm800k/prm800k/"
    math_train_path = os.path.join(base_dir, "math_splits", "train.jsonl")
    test_examples = ray.data.read_json(math_train_path)
    
    resources_kwarg: Dict[str, Any] = {}
    if tensor_parallel_size == 1:
        # For tensor_parallel_size == 1, we simply set num_gpus=1.
        resources_kwarg["num_gpus"] = 1
    else:
        # Otherwise, we have to set num_gpus=0 and provide
        # a function that will create a placement group for
        # each instance.
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

    ds_repeated = test_examples.flat_map(lambda x: [x] * 20)
    # Apply batch inference for all input data.
    ds = ds_repeated.map_batches(
        LLMPredictor,
        # Set the concurrency to the number of LLM instances.
        concurrency=num_instances,
        # Specify the batch size for inference.
        batch_size=128,
        **resources_kwarg,
    )

    output_path = os.path.join(base_dir, "sampling_data", "generator_data")
    ds.write_json(output_path)

