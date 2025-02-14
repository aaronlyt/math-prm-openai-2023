"""
make two type of dataset:
- smaller datasets (such as in phase 1 and the first few generations of phase 2) 
- larger datasets: contain all dataset?
"""
import os
import sys
import json
import re
import copy
import itertools
import torch as th
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from typing import Any, Dict, List
import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy



def parse_data(data_path, data_type, use_prod=False):
    """_summary_

    Args:
        data_path (_type_): _description_
        data_type (_type_): _description_
        use_prod (bool, optional): _description_. Defaults to False. 有多条路径只选择一条，True多个选项会进行组合

    Returns:
        _type_: _description_
    """
    datas = []
    trajs_count = 0
    step_count = 0
    truncstep_count = 0
    with open(data_path, 'r') as reader:
        for line in reader:
            line_item = json.loads(line)
            
            if line_item['is_quality_control_question'] or line_item['is_initial_screening_question']:
                continue
            
            if line_item['label']['finish_reason'] in ['bad_problem', 'give_up']:
                continue
            
            if data_type == 'small_data' and line_item['generation'] and line_item['generation'] > 4:
                continue
            
            example_ = {}
            example_['problem'] = line_item['question']['problem']
            example_['ground_truth_solution'] = line_item['question'].get('ground_truth_solution', None)
            example_['ground_truth_answer'] = line_item['question']['ground_truth_answer']
            
            labeled_solution = line_item['label']
            
            if not use_prod:
                prod_steps = [labeled_solution["steps"]]
            else:
                steps_ = [item['completions'] for item in labeled_solution["steps"]]
                print([len(item) for item in steps_])
                prod_steps = itertools.product(*steps_)
            
            for single_traj in prod_steps:
                trajs_count += 1
                step_count += len(single_traj)
                trajectories = []
                example_c = copy.deepcopy(example_)
                for step in single_traj:
                    truncstep_count += 1
                    if not use_prod:
                        if step["human_completion"] or step["chosen_completion"]:
                            chose_completion  = step["human_completion"] if step["chosen_completion"] is None \
                            else step["completions"][step["chosen_completion"]]
                        else:
                            chose_completion  = step["completions"][0]
                        # human rating will be null
                        if step["chosen_completion"] is None:
                            chose_completion['rating'] = 1
                        trajectories.append(chose_completion)
                    else:
                        chose_completion = step
                        # no human completion
                        trajectories.append(chose_completion)
                    
                    if chose_completion['rating'] == -1:
                        break
                # do not do any format work, leave it to train loop
                example_c['trajectories'] = trajectories
                
                datas.append(example_c)
                
    print(f"total steps count:{step_count}, trunc:{truncstep_count}, total trajectories count:{trajs_count}")
    return datas


if __name__ == '__main__':
    raw_datadir = '../../datasets_repo/prm800k/prm800k/data/'
    phrase1_data_path = os.path.join(raw_datadir, 'phase1_train.jsonl')
    phrase2_data_path = os.path.join(raw_datadir, 'phase2_train.jsonl')
    
    """
    data_type = 'small_data'
    sml_phrase1_data = parse_data(phrase1_data_path, data_type)
    sml_phrase2_data = parse_data(phrase2_data_path, data_type)
    small_data = sml_phrase1_data + sml_phrase2_data
    print('----total small data length is---', len(small_data))
    json.dump(small_data, open(os.path.join(raw_datadir, 'small_data.json'), 'w'))
    
    data_type = 'large_data'
    lrg_phrase1_data = parse_data(phrase1_data_path, data_type)
    lrg_phrase2_data = parse_data(phrase2_data_path, data_type)
    large_data = lrg_phrase1_data + lrg_phrase2_data
    print('----total larget data length is---', len(large_data))
    json.dump(large_data, open(os.path.join(raw_datadir, 'large_data.json'), 'w'))
    """
    
    data_type = 'large_data'
    # lrg_phrase1_data = parse_data(phrase1_data_path, data_type, use_prod=False)
    
    lrg_phrase2_data = parse_data(phrase2_data_path, data_type, use_prod=True)
    print('----total larget data length is---', len(lrg_phrase2_data))
    json.dump(lrg_phrase2_data, open(os.path.join(raw_datadir, 'large_ph2prod_data.json'), 'w'))
    
    lrg_phrase1_data = parse_data(phrase1_data_path, data_type, use_prod=False)
    large_prod_data = lrg_phrase1_data + lrg_phrase2_data
    print('----total larget data length is---', len(large_prod_data), len(lrg_phrase1_data))
    json.dump(large_prod_data, open(os.path.join(raw_datadir, 'large_prod_data.json'), 'w'))
    
    # ----total small data length is--- 26564
    # ----total larget data length is--- 74791