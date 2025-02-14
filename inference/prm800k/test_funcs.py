
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
import random
import re
import numpy as np
import torch as th
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

"""
using problem in MATH
few shot examplesfrom PRM
"""

def test_vllm_gen():
    model_path = "/t9k/mnt/workspace/llm_models/qwen2.5_1.5b/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    llm = LLM(model=model_path, trust_remote_code=True, max_model_len=8192, \
        tensor_parallel_size=1, enable_prefix_caching=True, enable_chunked_prefill=False)
    few_shots = [
        {"problem": "How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?", "solution": "The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$.  Therefore, the graph has $\\boxed{2}$ vertical asymptotes."},
        {"problem": "Compute\n\\[e^{2 \\pi i/13} + e^{4 \\pi i/13} + e^{6 \\pi i/13} + \\dots + e^{24 \\pi i/13}.\\]", "solution": "Let $\\omega = e^{2 \\pi i/13}.$  Then from the formula for a geometric sequence,\n\\begin{align*}\ne^{2 \\pi i/13} + e^{4 \\pi i/13} + e^{6 \\pi i/13} + \\dots + e^{24 \\pi i/13} &= \\omega + \\omega^2 + \\omega^3 + \\dots + \\omega^{12} \\\\\n&= \\omega (1 + \\omega + \\omega^2 + \\dots + \\omega^{11}) \\\\\n&= \\omega \\cdot \\frac{1 - \\omega^{12}}{1 - \\omega} \\\\\n&= \\frac{\\omega - \\omega^{13}}{1 - \\omega}.\n\\end{align*}Since $\\omega^{13} = (e^{2 \\pi i/13})^{13} = e^{2 \\pi i} = 1,$\n\\[\\frac{\\omega - \\omega^{13}}{1 - \\omega} = \\frac{\\omega - 1}{1 - \\omega} = \\boxed{-1}.\\]"},
        {"problem": "Let $S$ be the set of 10-tuples $(a_0, a_1, \\dots, a_9),$ where each entry is 0 or 1, so $S$ contains $2^{10}$ 10-tuples.  For each 10-tuple $s = (a_0, a_1, \\dots, a_9)$ in $S,$ let $p_s(x)$ be the polynomial of degree at most 9 such that\n\\[p_s(n) = a_n\\]for $0 \\le n \\le 9.$  For example, $p(x) = p_{(0,1,0,0,1,0,1,0,0,0)}(x)$ is the polynomial of degree at most 9 such that $p(0) = p(2) = p(3) = p(5) = p(7) = p(8) = p(9) = 0$ and $p(1) = p(4) = p(6) = 1.$\n\nFind\n\\[\\sum_{s \\in S} p_s(10).\\]", "solution": "Let\n\\[p(x) = \\sum_{s \\in S} p_s(x).\\]Then for any $n,$ $0 \\le n \\le 9,$\n\\[p(n) = \\sum_{s \\in S} p_s(n) = 2^9 = 512,\\]because $p_s(n) = 0$ for 512 polynomials $p_s(x),$ and $p_s(n) = 1$ for 512 polynomials $p_s(x).$\n\nThus, $p(x) = 512$ for 10 different values $n = 0,$ 1, 2, $\\dots,$ 9.  Also, $p(x)$ has degree at most 9.  Therefore, by the Identity Theorem, $p(x) = 512$ for all $x.$  In particular, $p(10) = \\boxed{512}.$"}
    ]
    fewshot_str = "\n\n".join([f"<|im_start|> ### problem: {fs['problem']} ### solution: {fs['solution']} <|im_end|>" for fs in few_shots])

    # Create a class to do batch inference. Create a sampling params object.
    endoftoken = tokenizer.eos_token_id
    stop_token_ids = [endoftoken, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    sampling_params = SamplingParams(temperature=0.6, top_p=1, max_tokens=1024, stop_token_ids=stop_token_ids)

    datas = []
    base_dir = "../datasets_repo/prm800k/prm800k/"
    math_train_path = os.path.join(base_dir, "math_splits", "train.jsonl")
    with open(math_train_path, 'r') as f:
        for line in f:
            datas.append(json.loads(line))
    batch = {"problem": "In triangle $ABC,$ $E$ lies on $\\overline{AC}$ such that $AE:EC = 2:1,$ and $F$ lies on $\\overline{AB}$ such that $AF:FB = 1:4.$  Let $P$ be the intersection of $\\overline{BE}$ and $\\overline{CF}.$\n\n[asy]\nunitsize(0.8 cm);\n\npair A, B, C, D, E, F, P;\n\nA = (1,4);\nB = (0,0);\nC = (6,0);\nE = interp(A,C,2/3);\nF = interp(A,B,1/5);\nP = extension(B,E,C,F);\n\ndraw(A--B--C--cycle);\ndraw(B--E);\ndraw(C--F);\n\nlabel(\"$A$\", A, N);\nlabel(\"$B$\", B, SW);\nlabel(\"$C$\", C, SE);\nlabel(\"$E$\", E, NE);\nlabel(\"$F$\", F, W);\nlabel(\"$P$\", P, S);\n[/asy]\n\nThen\n\\[\\overrightarrow{P} = x \\overrightarrow{A} + y \\overrightarrow{B} + z \\overrightarrow{C},\\]where $x,$ $y,$ and $z$ are constants such that $x + y + z = 1.$  Enter the ordered triple $(x,y,z).$", "solution": "From the given information,\n\\[\\overrightarrow{E} = \\frac{1}{3} \\overrightarrow{A} + \\frac{2}{3} \\overrightarrow{C}\\]and\n\\[\\overrightarrow{F} = \\frac{4}{5} \\overrightarrow{A} + \\frac{1}{5} \\overrightarrow{B}.\\]Isolating $\\overrightarrow{A}$ in each equation, we obtain\n\\[\\overrightarrow{A} = 3 \\overrightarrow{E} - 2 \\overrightarrow{C} = \\frac{5 \\overrightarrow{F} - \\overrightarrow{B}}{4}.\\]Then $12 \\overrightarrow{E} - 8 \\overrightarrow{C} = 5 \\overrightarrow{F} - \\overrightarrow{B},$ so $12 \\overrightarrow{E} + \\overrightarrow{B} = 5 \\overrightarrow{F} + 8 \\overrightarrow{C},$ or\n\\[\\frac{12}{13} \\overrightarrow{E} + \\frac{1}{13} \\overrightarrow{B} = \\frac{5}{13} \\overrightarrow{F} + \\frac{8}{13} \\overrightarrow{C}.\\]Since the coefficients on both sides of the equation add up to 1, the vector on the left side lies on line $BE,$ and the vector on the right side lies on line $CF.$  Therefore, this common vector is $\\overrightarrow{P}.$  Then\n\\begin{align*}\n\\overrightarrow{P} &= \\frac{12}{13} \\overrightarrow{E} + \\frac{1}{13} \\overrightarrow{B} \\\\\n&= \\frac{12}{13} \\left( \\frac{1}{3} \\overrightarrow{A} + \\frac{2}{3} \\overrightarrow{C} \\right) + \\frac{1}{13} \\overrightarrow{B} \\\\\n&= \\frac{4}{13} \\overrightarrow{A} + \\frac{1}{13} \\overrightarrow{B} + \\frac{8}{13} \\overrightarrow{C}.\n\\end{align*}Thus, $(x,y,z) = \\boxed{\\left( \\frac{4}{13}, \\frac{1}{13}, \\frac{8}{13} \\right)}.$", "answer": "\\left( \\frac{4}{13}, \\frac{1}{13}, \\frac{8}{13} \\right)", "subject": "Precalculus", "level": 5, "unique_id": "train/precalculus/422.json"}
    prompt = f"{fewshot_str} <|im_start|> ### problem: {batch['problem']} \n### solution: "
    print(prompt)
    outputs = llm.generate(prompt, sampling_params)


def test_vllm_genv2():
    model_path = "/t9k/mnt/workspace/llm_models/qwen2.5_1.5b/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    llm = LLM(model=model_path, trust_remote_code=True, max_model_len=8192, \
        tensor_parallel_size=1, enable_prefix_caching=True, enable_chunked_prefill=False)
    few_shots = []
    large_data = json.load(open("../datasets_repo/prm800k/prm800k/data/large_data.json", "r"))
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
    few_shots_ = random.sample(few_shots, 3)
    fewshot_str = "\n\n\n".join([f"problem: {fs['problem']}\nsolution: {fs['solution']}" for fs in few_shots_]).strip(" ")

    
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "\n\n\nproblem:"]

    # Create a class to do batch inference. Create a sampling params object.
    stop_token_ids = [151645, 151643]
    sampling_params = SamplingParams(temperature=0.6, top_p=1, max_tokens=1024, stop_token_ids=stop_token_ids, stop=stop_words)

    datas = []
    base_dir = "../datasets_repo/prm800k/prm800k/"
    math_train_path = os.path.join(base_dir, "math_splits", "train.jsonl")
    with open(math_train_path, 'r') as f:
        for line in f:
            datas.append(json.loads(line))
    batch = {"problem": "In triangle $ABC,$ $E$ lies on $\\overline{AC}$ such that $AE:EC = 2:1,$ and $F$ lies on $\\overline{AB}$ such that $AF:FB = 1:4.$  Let $P$ be the intersection of $\\overline{BE}$ and $\\overline{CF}.$\n\n[asy]\nunitsize(0.8 cm);\n\npair A, B, C, D, E, F, P;\n\nA = (1,4);\nB = (0,0);\nC = (6,0);\nE = interp(A,C,2/3);\nF = interp(A,B,1/5);\nP = extension(B,E,C,F);\n\ndraw(A--B--C--cycle);\ndraw(B--E);\ndraw(C--F);\n\nlabel(\"$A$\", A, N);\nlabel(\"$B$\", B, SW);\nlabel(\"$C$\", C, SE);\nlabel(\"$E$\", E, NE);\nlabel(\"$F$\", F, W);\nlabel(\"$P$\", P, S);\n[/asy]\n\nThen\n\\[\\overrightarrow{P} = x \\overrightarrow{A} + y \\overrightarrow{B} + z \\overrightarrow{C},\\]where $x,$ $y,$ and $z$ are constants such that $x + y + z = 1.$  Enter the ordered triple $(x,y,z).$", "solution": "From the given information,\n\\[\\overrightarrow{E} = \\frac{1}{3} \\overrightarrow{A} + \\frac{2}{3} \\overrightarrow{C}\\]and\n\\[\\overrightarrow{F} = \\frac{4}{5} \\overrightarrow{A} + \\frac{1}{5} \\overrightarrow{B}.\\]Isolating $\\overrightarrow{A}$ in each equation, we obtain\n\\[\\overrightarrow{A} = 3 \\overrightarrow{E} - 2 \\overrightarrow{C} = \\frac{5 \\overrightarrow{F} - \\overrightarrow{B}}{4}.\\]Then $12 \\overrightarrow{E} - 8 \\overrightarrow{C} = 5 \\overrightarrow{F} - \\overrightarrow{B},$ so $12 \\overrightarrow{E} + \\overrightarrow{B} = 5 \\overrightarrow{F} + 8 \\overrightarrow{C},$ or\n\\[\\frac{12}{13} \\overrightarrow{E} + \\frac{1}{13} \\overrightarrow{B} = \\frac{5}{13} \\overrightarrow{F} + \\frac{8}{13} \\overrightarrow{C}.\\]Since the coefficients on both sides of the equation add up to 1, the vector on the left side lies on line $BE,$ and the vector on the right side lies on line $CF.$  Therefore, this common vector is $\\overrightarrow{P}.$  Then\n\\begin{align*}\n\\overrightarrow{P} &= \\frac{12}{13} \\overrightarrow{E} + \\frac{1}{13} \\overrightarrow{B} \\\\\n&= \\frac{12}{13} \\left( \\frac{1}{3} \\overrightarrow{A} + \\frac{2}{3} \\overrightarrow{C} \\right) + \\frac{1}{13} \\overrightarrow{B} \\\\\n&= \\frac{4}{13} \\overrightarrow{A} + \\frac{1}{13} \\overrightarrow{B} + \\frac{8}{13} \\overrightarrow{C}.\n\\end{align*}Thus, $(x,y,z) = \\boxed{\\left( \\frac{4}{13}, \\frac{1}{13}, \\frac{8}{13} \\right)}.$", "answer": "\\left( \\frac{4}{13}, \\frac{1}{13}, \\frac{8}{13} \\right)", "subject": "Precalculus", "level": 5, "unique_id": "train/precalculus/422.json"}
    problem = batch['problem'].replace('\n', ' ')
    prompt = f"{fewshot_str}\n\n\nproblem: {problem}\nsolution:"
    print(prompt)
    outputs = llm.generate(prompt, sampling_params)
    print(outputs[0].outputs[0])
    
    
def  test_vllm_genv3():
    model_path =  "../scripts/tmp/prm_prmlge/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    llm = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation = "flash_attention_2", \
            device_map="auto", torch_dtype=th.float16)
    device = th.device("cuda")
    
    base_dir = "../datasets_repo/prm800k/prm800k/"
    test_directory = os.path.join(base_dir, "sampling_data", "mathte_sftgen_samps")
    sel_test_path = os.path.join(test_directory, "2_000019_000000.json")
    te_data = []
    with open(sel_test_path, "r") as f:
        for line in f:
            te_data.append(json.loads(line))
    batch = te_data[:4]
    problems = ["### Problem: " + ex["problem"] + "\n ### Solution: " for ex in batch]
    # solutions = [u'\n'.join(['<START_STEP>'+ re.sub(r'STEP \d+:\s*', '', x).strip() + '<END_STEP>' for x in ex["solution_sample"].split('\n')]) for ex in batch]
    solutions = []
    for sol in batch:
        sol_ = []
        for step in sol["solution_sample"].split('\n'):
            step_ = '<START_STEP>' + re.sub(r'STEP \d+:\s*', '', step).strip() + '<END_STEP>'
            sol_.append(step_)
            if 'The answer is' in step:
                break
        solutions.append(u'\n'.join(sol_))
    batch_inputs_pair = [problem + solution for problem, solution in zip(problems, solutions)]

    print("----batch inputs pair----", batch_inputs_pair[0])
    
    inputs_repr = tokenizer(batch_inputs_pair, \
        padding=True, truncation=True, return_tensors="pt", max_length=1024, add_special_tokens=False)
    inputs_repr = {k: v.to(device) for k, v in inputs_repr.items()}
    token_reward_logits = llm(**inputs_repr).logits.detach().cpu().numpy()
    
    endofstep_idx = tokenizer.convert_tokens_to_ids("<END_STEP>")
    print('---endofstep idx---', endofstep_idx)
    batch_endofstep_ids = []
    for i in range(len(batch_inputs_pair)):
        print('---end of newline positions--', np.where(inputs_repr["input_ids"][i].detach().cpu().numpy() == endofstep_idx)[0])
        batch_endofstep_ids.append(np.where(inputs_repr["input_ids"][i].detach().cpu().numpy() == endofstep_idx)[0])
    
    reward_logits_sum = []
    for idx in range(len(batch_inputs_pair)):
        endofstep_ids = batch_endofstep_ids[idx]
        print(token_reward_logits[idx, endofstep_ids, endofstep_idx])
        reward_logits_sum.append(np.log(token_reward_logits[idx, endofstep_ids, endofstep_idx]).sum())
    print("reward_logits_sum", reward_logits_sum)


def test_vllm_genv6():
    model_path = "/t9k/mnt/workspace/llm_models/qwen2.5_1.5b/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    llm = LLM(model=model_path, trust_remote_code=True, max_model_len=8192, \
        tensor_parallel_size=1, enable_prefix_caching=True, enable_chunked_prefill=False)
    
    base_dir = "../datasets_repo/prm800k/prm800k/"
    test_directory = os.path.join(base_dir, "sampling_data", "mathte_sftgen_samps")
    sel_test_path = os.path.join(test_directory, "2_000019_000000.json")
    te_data = []
    with open(sel_test_path, "r") as f:
        for line in f:
            te_data.append(json.loads(line))
    batch = te_data[0]
    problem = ["### Problem: " + batch["problem"] + "\n ### Solution: "][0]
    part_solutions = []
    cur_steps = problem
    for step in batch["solution_sample"].split('\n'):
        cur_steps = cur_steps + "\n" + step
        part_solutions.append(cur_steps + '\n')
    
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "\n\n\nproblem:"]

    # Create a class to do batch inference. Create a sampling params object.
    stop_token_ids = [151645, 151643]
    sampling_params = SamplingParams(temperature=0.6, n=16, max_tokens=1024, \
            stop_token_ids=stop_token_ids, stop=stop_words)
    
    outputs = llm.generate(part_solutions, sampling_params)

    import IPython
    IPython.embed()
    
if __name__ == "__main__":
    test_vllm_genv6()