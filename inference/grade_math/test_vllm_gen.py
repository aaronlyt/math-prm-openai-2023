import json
import numpy as np
import torch as th
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    
    test_path = "../../datasets_repo/grade-school-math/grade_school_math/verify_data/test_samples_4verify/2_000000_000000.json"
    test_data = []
    with open(test_path, 'r') as reader:
        for line in reader:
            test_data.append(json.loads(line))
    batch = test_data[:8]
    batch_raw_inputs = [item['answer_sample'] for item in batch]
    
    model_path = "../scripts/tmp/verify_model/checkpoint-000/"
    
    # not works: ValueError: allowed_token_ids contains out-of-vocab token id(using vocab size, vocab.json); vocab size is not same with tokenizer size,
    """
    llm = LLM(model=model_path,  trust_remote_code=True, max_model_len=512, \
        tensor_parallel_size=1, enable_prefix_caching=True, enable_chunked_prefill=False)
    sampling_params = SamplingParams(temperature=0.6, top_p=1, max_tokens=1, \
        logprobs=True, allowed_token_ids=[151665], skip_special_tokens=False)
    print(llm.generate(batch_qn_prompt[1], sampling_params))
    """
    device = th.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer.padding_side = "left"
    inputs_repr = tokenizer(batch_raw_inputs, \
        padding=True, truncation=True, return_tensors="pt", max_length=1024, add_special_tokens=False)
    
    inputs_repr = {k: v.to(device) for k, v in inputs_repr.items()}
    token_reward_logits = llm(**inputs_repr).logits.detach().cpu().numpy()
    
    response_template = " ### Answer: "
    response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
        
    batch_anstoken_sidx = []
    for i in range(len(batch_raw_inputs)):
        for assistant_idx in np.where(inputs_repr["input_ids"][i].detach().cpu().numpy() == response_token_ids[0])[0]:
            print(response_token_ids, inputs_repr["input_ids"][i][assistant_idx : assistant_idx + len(response_token_ids)].tolist())
            # find the indexes of the start of a response.
            if (
                response_token_ids == inputs_repr["input_ids"][i][assistant_idx : assistant_idx + len(response_token_ids)].tolist()
            ):
                batch_anstoken_sidx.append(assistant_idx)
    print('--------', batch_anstoken_sidx)
    reward_logits_sum = []
    for idx in range(len(batch_raw_inputs)):
        s_len = batch_anstoken_sidx[idx] + (1 - inputs_repr["attention_mask"][idx]).sum()
        print(token_reward_logits[idx, s_len:, -1].shape[0], len(inputs_repr["input_ids"][idx]))
        print(token_reward_logits[idx, s_len:, -1])
        reward_logits_sum.append(token_reward_logits[idx, s_len:, -1].sum())
