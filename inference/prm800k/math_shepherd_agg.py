"""
aggregate the math-shepherd sampling dataset, execute after math_shepherd_data
"""
import os
import sys
import json
import glob
from typing import Any, Dict, List
import numpy as np
import pandas as pd
# import fireducks.pandas as pd


if __name__ == "__main__":
    base_dir = "../../datasets_repo/prm800k/"
    shepherd_datas = []
    for subdir in ['shepherd_annotation_steps_data', 'shepherd_annotation_steps_data_attempt2']:
        shepherd_dir = os.path.join(base_dir, "prm800k/shepsherd_data", subdir)
        for dname in glob.glob(os.path.join(shepherd_dir, "*.json")):
            with open(os.path.join(shepherd_dir, dname), "r") as reader:
                for line in reader:
                    line_data = json.loads(line)
                    shepherd_datas.append([line_data['problem'], line_data['solution_sample'], line_data['answer'], \
                        line_data['step_idx'], line_data['step_solution'], line_data['steps_hard_rating'], line_data['steps_soft_rating']])
    
    print('----step dataset length is----', len(shepherd_datas))
    shepherd_pddatas = pd.DataFrame(shepherd_datas, columns=['problem', 'solution_sample', 'answer', 'step_idx', 'step_solution', 'hard_rating', 'soft_rating'])
    stepagg_datas = []
    stepagg_steps = 0
    to_path = os.path.join(base_dir, 'shepsherd_data/shepherd_annsteps.json')
    with open(to_path, 'w') as writer:
        for com_key, gp_data in shepherd_pddatas.groupby(['problem', 'solution_sample']):
            problem, solution_sample = com_key
            gp_data = gp_data.sort_values(['step_idx'])
            hard_ratings = gp_data['hard_rating'].to_list()
            soft_ratings = gp_data['soft_rating'].to_list()
            # contains the whole solution
            steps_ppsl = gp_data['step_solution'].to_list()[-1]
            steps = steps_ppsl.split('### Solution: \n')[1]
            if steps.endswith('\n'):
                steps = steps[:-2] # remove the last \n
            
            step_count = len(steps.split('\n'))
            if step_count != len(hard_ratings):
                # print('---step count and rating count not equal---', steps)
                # print(hard_ratings)
                continue
            answer = gp_data['answer'].to_list()[0]
            stepagg_datas.append([problem, answer, steps, hard_ratings, soft_ratings])
            stepagg_steps += step_count
            writer.write(json.dumps({'problem': problem, 'answer': answer, 'steps': steps, \
                'hard_ratings': hard_ratings, 'soft_ratings': soft_ratings}) + '\n')
    
    print('---total solution sample level dataset length is ---', len(stepagg_datas), stepagg_steps)