"""
generating training dataset using: MATH-SHEPHERD: VERIFY AND REINFORCE LLMS
STEP-BY-STEP WITHOUT HUMAN ANNOTATIONS
paramter setting:  D. PRM Training Details
shepsherd_steps.py
math_shepherd_data.py
math_shepherd_agg.py
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
import re
import copy
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

from grader import grade_answer


# https://docs.vllm.ai/en/stable/getting_started/examples/offline_inference_distributed.html
# Create a class to do batch inference.
# Set tensor parallelism per instance.
tensor_parallel_size = 1
# Set number of instances. Each instance will use tensor_parallel_size GPUs.
num_instances = 4


class LLMRanker:
    def __init__(self):
        # Create an LLM.
        model_path = "../../scripts/tmp/prm_sftgen_1e-5/checkpoint-9554/"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.llm = LLM(model=model_path, trust_remote_code=True, max_model_len=2048, \
            tensor_parallel_size=tensor_parallel_size, enable_prefix_caching=True, enable_chunked_prefill=False)
        
        # Create a class to do batch inference. Create a sampling params object.
        # step end token? \n?, STEP?
        stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "### Problem: "]
        stop_token_ids = [151645, 151643]
        self.sampling_params = SamplingParams(temperature=0.7, n=16, max_tokens=1024, \
            stop_token_ids=stop_token_ids, stop=stop_words)
    
    
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        """
        Args:
            batch (Dict[str, np.ndarray]): _description_

        Returns:
            Dict[str, list]: _description_
        """
        outputs = self.llm.generate(batch['step_solution'], self.sampling_params)
        
        batch_samples = copy.deepcopy(batch)
        batch_samples['steps_hard_rating'] = []
        batch_samples['steps_soft_rating'] = []
        
        for batch_idx, step_mcts_outputs in enumerate(outputs):
            exist_corr = 0
            corr_counnt = 0
            for mcts_output in step_mcts_outputs.outputs:
                if mcts_output.finish_reason != "stop":
                    continue
                pred_answer = None
                for cand_ansline in mcts_output.text.split('\n'):
                    if 'The answer is:' in cand_ansline:
                        pred_answer = cand_ansline.split("The answer is:")[1].strip()
                        break
                if not pred_answer:
                    continue
                
                if not grade_answer(pred_answer, batch['answer'][batch_idx]):
                    continue
                
                exist_corr = 1
                corr_counnt += 1

            batch_samples['steps_hard_rating'].append(exist_corr)
            batch_samples['steps_soft_rating'].append(corr_counnt / 16.0)
        
        # Run inference.
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
    steps_path = os.path.join("../datasets_repo/prm800k/shepsherd_data/", "mathtra80_sftgen1e-5_steps")
    test_examples = ray.data.read_json(steps_path)
    
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
        batch_size=256,
        **resources_kwarg,
    )

    output_path = os.path.join(base_dir, "shepsherd_data", "stepann_samp80")
    ds.write_json(output_path)
    
    # Peek first 10 results.
    # NOTE: This is for local testing and debugging. For production use case,
    # one should write full result out as shown below.
    outputs = ds.take(limit=2)
    for output in outputs:
        print(output)

