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
num_instances = 4

class LLMPredictor:
    def __init__(self):
        # Create an LLM.
        model_path = "../../scripts/tmp/checkpoint-624"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.llm = LLM(model=model_path, trust_remote_code=True, max_model_len=1024, \
            tensor_parallel_size=tensor_parallel_size, enable_prefix_caching=True, enable_chunked_prefill=False)
        self.sample_len = 100

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        EQUALS_TOKENS = set([28, 284, 11730])
        last_answers = [-1000] * len(batch['question'])
        batch_qn = batch['question']
        batch_qn_prompt = [f"### Question: {qn}\n ### Answer: " for qn in batch_qn] 
        
        for _ in range(self.sample_len):
            outputs = self.llm.generate(batch_qn_prompt, sampling_params)
            prompts: List[str] = []
            generated_text: List[str] = []
            for output in outputs:
                prompts.append(output.prompt)
                generated_text.append(''.join([o.text for o in output.outputs]))

            qns = [q + t for q, t in zip(prompts, generated_text)]

            # Check if any of the new tokens are in the EQUALS_TOKENS set
            for i, tok in enumerate(generated_text):
                if self.tokenizer.encode(tok) and self.tokenizer.encode(tok)[0] in EQUALS_TOKENS:
                    answer = use_calculator(qns[i])
                    if answer is not None:
                        print(f"Triggered calculator, answer {answer}")
                        qns[i] += str(answer) + ">>"
                        last_answers[i] = answer
            
            batch_qn_prompt = qns
            # print('---last_answers---', last_answers)
        # get the label
        batch_sample_labels = []
        for answer, qn_answer in zip(last_answers, batch["answer"]):
            try:
                qn_answer = eval(qn_answer.replace("<|endoftext|>", "").split("#### ")[-1].strip().replace(",", ""))
                batch_sample_labels.append(int(abs(answer - qn_answer) < 1e-6))
            except Exception as e:
                batch_sample_labels.append(0)
                print(f"Warning: Failed to eval {answer}, answer: {qn_answer}, exception: {e}")

        # print("batch_sample_labels", batch_sample_labels)
        return {
            "question": batch['question'],
            "raw_answer": batch['answer'],
            "answer_sample": batch_qn_prompt,
            "batch_sample_label": batch_sample_labels
        }

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
    test_path = os.path.join(base_dir, "data", "test.jsonl")
    test_examples = ray.data.read_json(test_path)
    
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

    ds_repeated = test_examples.flat_map(lambda x: [x] * 100)
    # Apply batch inference for all input data.
    ds = ds_repeated.map_batches(
        LLMPredictor,
        # Set the concurrency to the number of LLM instances.
        concurrency=num_instances,
        # Specify the batch size for inference.
        batch_size=64,
        **resources_kwarg,
    )

    # Peek first 10 results.
    # NOTE: This is for local testing and debugging. For production use case,
    # one should write full result out as shown below.
    outputs = ds.take(limit=2)
    for output in outputs:
        prompt = output["batch_sample_label"]
        generated_text = output["answer_sample"]
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
    output_path = os.path.join(base_dir, "verify_data", "test_samples_4verify")
    ds.write_json(output_path)
