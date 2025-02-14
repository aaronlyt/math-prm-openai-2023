import os
import sys
from contextlib import contextmanager
import signal
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tfdataset.gradesm_data import get_examples

# taken from
# https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            return eval(formula)
    except Exception as e:
        signal.alarm(0)
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


def sample(model, qn, tokenizer, device, sample_len):
    # Inefficient version of calculator sampling -- no batches, doesn't
    # cache activations from previous tokens
    #symbols: =, Ġ=, )=; tokens unchanged, index changed as tokenizer change
    EQUALS_TOKENS = set([28, 284, 11730])
    answer = None
    
    for _ in range(sample_len):
        with torch.no_grad():
            toks = tokenizer([qn], padding=False, return_tensors="pt").to(device)
            # orig_len = toks["input_ids"].shape[1]

            out = model.generate(
                **toks, max_new_tokens=1, pad_token_id=model.config.eos_token_id
            )
            text = tokenizer.batch_decode(out)[0]
            
            if out[0, -1].item() in EQUALS_TOKENS:
                answer = use_calculator(text)
                if answer is not None:
                    print("Triggered calculator, answer", answer)
                    text = text + str(answer) + ">>"

            qn = text
    return qn, answer


def batch_sample(model, qns, tokenizer, device, sample_len):
    # Symbols that indicate the end of the question and start of the answer
    EQUALS_TOKENS = set([28, 284, 11730])
    last_answers = [-1e6] * len(qns)
    tokenizer.padding_side = "left"
    past_key_values = None  # Cache for storing past key values
    for _ in range(sample_len):
        with torch.no_grad():
            toks = tokenizer(qns, padding=True, truncation=True, return_tensors="pt").to(device)

            out = model.generate(
                **toks,
                max_new_tokens=1,
                pad_token_id=model.config.eos_token_id,
                # past_key_values=past_key_values,
                # use_cache=True
            )

            # Extract the last token's output for each sequence in the batch
            new_tokens = out[:, -1:]
            new_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            # Update the questions with the newly generated tokens
            qns = [q + t for q, t in zip(qns, new_texts)]

            # Check if any of the new tokens are in the EQUALS_TOKENS set
            for i, tok in enumerate(new_tokens.squeeze(-1)):
                if tok.item() in EQUALS_TOKENS:
                    answer = use_calculator(qns[i])
                    if answer is not None:
                        print(f"Triggered calculator, answer {answer}")
                        qns[i] += str(answer) + ">>"
                        last_answers[i] = answer
                
            # Update the past key values for the next iteration
            # past_key_values = model.past_key_values()

    return qns, last_answers

# Note: The function `use_calculator` is assumed to be defined elsewhere.

if __name__ == "__main__":
    model_path = '../../scripts/tmp/checkpoint-4680'
    # model_path = '/t9k/mnt/workspace/llm_models/qwen2.5_1.5b' # no corrected item
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    device = torch.device("cuda")
    model.to(device)

    test_examples = get_examples("test")
    
    sample_len = 100
    count = 0
    batch_size = 128
    batch_count = len(test_examples) // batch_size + 1
    for batch_idx in range(batch_count):
        batch_qn = test_examples[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_qn_prompt = [f"### Question: {qn['question']}\n ### Answer: " for qn in batch_qn] 
        batch_out, batch_answers = batch_sample(model, batch_qn_prompt, tokenizer, device, sample_len)
        batch_qn_answer = [eval(qn["answer"].replace("<|endoftext|>", "").split("#### ")[-1].strip()) for qn in batch_qn]
        
        for answer, qn_answer in zip(batch_answers, batch_qn_answer):
            try:
                count += abs(answer - qn_answer) < 1e-6     
            except Exception as e:
                print(f"Warning: Failed to eval {answer}, answer: {qn_answer}, exception: {e}")
    # 0.2721
    print(count / len(test_examples))