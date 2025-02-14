import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import sys
import re
import scipy
import torch as th
from contextlib import contextmanager
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import Any, Dict, List
import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, SamplingParams


# https://docs.vllm.ai/en/stable/getting_started/examples/offline_inference_distributed.html
# Create a class to do batch inference.
# Set tensor parallelism per instance.
tensor_parallel_size = 1
# Set number of instances. Each instance will use tensor_parallel_size GPUs.
num_instances = 4


class LLMRanker:
    def __init__(self):
        # Create an LLM.
        model_path = "../../scripts/tmp/woembdecay_ep5_shepsherd_8e-05/checkpoint-1050/"
        
        self.tokenizer = AutoTokenizer.from_pretrained("../scripts/tmp/woembdecay_ep5_shepsherd_8e-05/")
        self.tokenizer.padding_side = "left"
        self.llm = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation = "flash_attention_2", \
                device_map="auto", torch_dtype=th.float16)
        self.llm.eval()
        self.device = th.device("cuda")
        
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        problems = ["### Problem: " + ex + "\n ### Solution: " for ex in batch["problem"]]
        solutions = []
        for sol in batch["solution_sample"]:
            sol_ = []
            for step in sol.split('\n'):
                step_ = '<START_STEP>' + re.sub(r'STEP \d+:\s*', '', step).strip() + '<END_STEP>'
                # sol_.append(step_)
                if 'The answer is' in step and len(sol_) >= 1:
                    break
                
                # another method, remove all answer steps for shepherd
                sol_.append(step_)
            solutions.append(u'\n'.join(sol_))
        batch_inputs_pair = [problem + solution for problem, solution in zip(problems, solutions)]
        
        with th.no_grad():
            inputs_repr = self.tokenizer(batch_inputs_pair, \
                padding=True, truncation=True, return_tensors="pt", max_length=1024)
            inputs_repr = {k: v.to(self.device) for k, v in inputs_repr.items()}
            token_reward_logits = self.llm(**inputs_repr).logits.detach().cpu().numpy()
        
        endofstep_idx = self.tokenizer.convert_tokens_to_ids("<END_STEP>")
        postep_idx = self.tokenizer.convert_tokens_to_ids("<POS_STEP>")
        negstep_idx = self.tokenizer.convert_tokens_to_ids("<NEG_STEP>")
        batch_endofstep_ids = []
        for i in range(len(batch_inputs_pair)):
            batch_endofstep_ids.append(np.where(inputs_repr["input_ids"][i].detach().cpu().numpy() == endofstep_idx)[0])
        reward_logits_sum = []
        for idx in range(len(batch_inputs_pair)):
            # using end of token: batch, len(endofstep_ids), vocab_size
            endofstep_ids = batch_endofstep_ids[idx]

            # reward_logits_sum.append(np.exp(token_reward_logits[idx, endofstep_ids, endofstep_idx]).sum())
            # previous exp are all wrong, should use <POS_STEP> probability, this, should re-test , no significant difference?
            # reward_logits_sum.append(np.min(token_reward_logits[idx, endofstep_ids, postep_idx))
            
            # TODO: softmax on postep_idx and negstep_id, using the probablity as min input
            pos_logits = token_reward_logits[idx, endofstep_ids, postep_idx]
            neg_logits = token_reward_logits[idx, endofstep_ids, negstep_idx]
            postep_score = scipy.special.softmax(np.stack([pos_logits, neg_logits], axis=1), axis=1)
            try:
            # using min step
                reward_logits_sum.append(np.min(postep_score[:, 0]))
            except:
                print(token_reward_logits.shape)
                print(batch["solution_sample"][idx])
                print(batch_inputs_pair[idx])
            # using last step
            # reward_logits_sum.append(postep_score[-1, 0])
            
            # for binary cls
            # reward_logits_sum.append(np.min(token_reward_logits[idx, endofstep_ids, endofstep_idx]))
        # print("batch_sample_labels", batch_sample_labels)
        batch["reward_logits_sum"] = reward_logits_sum
        return batch

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
    test_directory = os.path.join(base_dir, "sampling_data", "mathte_sftgen5e-6_samps")
    test_examples = ray.data.read_json(test_directory)
    
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
    
    # Apply batch inference for all input data.
    ds = test_examples.map_batches(
        LLMRanker,
        # Set the concurrency to the number of LLM instances.
        concurrency=num_instances,
        # Specify the batch size for inference.
        batch_size=64,
        **resources_kwarg,
    )

    output_path = os.path.join(base_dir, "sampling_data", "shepsherd_8e-05_ep10")
    ds.write_json(output_path)
    
    # Peek first 10 results.
    # NOTE: This is for local testing and debugging. For production use case,
    # one should write full result out as shown below.
    outputs = ds.take(limit=2)
    for output in outputs:
        print(output)
    
