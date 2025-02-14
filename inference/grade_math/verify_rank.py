import os
import sys
from contextlib import contextmanager
import signal
import torch as th
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import Any, Dict, List
import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from vllm import LLM, SamplingParams

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tfdataset.gradesm_data import get_examples


def eval_with_timeout(formula, max_time=3):
    try:
        return eval(formula)
    except Exception as e:
        print(f"Warning: Failed to eval {formula}, exception: {e}")
        return None


def use_calculator(sample):
    if "<<" not in sample:
        return None

    parts = sample.split("<<")
    remaining = parts[-1]
    if ">>" in remaining:
        return None
    if "=" not in remaining:
        return None
    lhs = remaining.split("=")[0]
    lhs = lhs.replace(",", "")
    if any([x not in "0123456789*+-/.()" for x in lhs]):
        return None
    return eval_with_timeout(lhs)


# https://docs.vllm.ai/en/stable/getting_started/examples/offline_inference_distributed.html
# Create a class to do batch inference.
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.6, top_p=1, max_tokens=1)
# Set tensor parallelism per instance.
tensor_parallel_size = 1
# Set number of instances. Each instance will use tensor_parallel_size GPUs.
num_instances = 3


class LLMRanker:
    def __init__(self):
        # Create an LLM.
        model_path = "../../scripts/tmp/verify_model/checkpoint-000/"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        self.llm = AutoModelForCausalLM.from_pretrained(model_path, \
            attn_implementation = "flash_attention_2", \
                device_map="auto", torch_dtype=th.float16)
        self.device = th.device("cuda")
        
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        batch_raw_inputs = [item for item in batch['answer_sample']]
        inputs_repr = self.tokenizer(batch_raw_inputs, \
            padding=True, truncation=True, return_tensors="pt", max_length=1024, add_special_tokens=False)
        inputs_repr = {k: v.to(self.device) for k, v in inputs_repr.items()}
        token_reward_logits = self.llm(**inputs_repr).logits.detach().cpu().numpy()
        
        response_template = " ### Answer: "
        response_token_ids = self.tokenizer.encode(response_template, add_special_tokens=False)
        batch_anstoken_sidx = []
        for i in range(len(batch_raw_inputs)):
            for assistant_idx in np.where(inputs_repr["input_ids"][i].detach().cpu().numpy() == response_token_ids[0])[0]:
                # print(response_token_ids, inputs_repr["input_ids"][i][assistant_idx : assistant_idx + len(response_token_ids)].tolist())
                # find the indexes of the start of a response.
                if (
                    response_token_ids == inputs_repr["input_ids"][i][assistant_idx : assistant_idx + len(response_token_ids)].tolist()
                ):
                    batch_anstoken_sidx.append(assistant_idx)
        reward_logits_sum = []
        for idx in range(len(batch_raw_inputs)):
            s_len = batch_anstoken_sidx[idx] + (1 - inputs_repr["attention_mask"][idx]).sum()
            # print(np.exp(token_reward_logits[idx, s_len:, -1]).sum())
            reward_logits_sum.append(np.exp(token_reward_logits[idx, s_len:, -1]).sum())
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
    base_dir = "../../datasets_repo/grade-school-math/grade_school_math/"
    test_directory = os.path.join(base_dir, "verify_data", "test_samples_4verify")
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
        batch_size=32,
        **resources_kwarg,
    )

    output_path = os.path.join(base_dir, "verify_data", "verify_ranker")
    ds.write_json(output_path)
    
    # Peek first 10 results.
    # NOTE: This is for local testing and debugging. For production use case,
    # one should write full result out as shown below.
    outputs = ds.take(limit=2)
    for output in outputs:
        prompt = output["batch_sample_label"]
        generated_text = output["answer_sample"]
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

