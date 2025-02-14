import os
import sys
import re
import copy
import glob
import json
from typing import Any, Dict, List


def construct_steps_data():
    base_dir = "../../datasets_repo/prm800k/"
    samp16_dir = os.path.join(base_dir, "shepsherd_data", "mathtra80_sftgen1e-5_samps")
    # mathtra80_sftgen1e-5_steps
    steps_path_dir = os.path.join("../../datasets_repo/prm800k/shepsherd_data/", "mathtra80_sftgen1e-5_steps")
    steps_count = 0
    solution_count = 0
    writer = open(os.path.join(steps_path_dir, "0.json"), "w")
    write_count = 0
    read_filecount = 0
    for dname in glob.glob(os.path.join(samp16_dir , "*.json")):
        read_filecount += 1
        with open(os.path.join(samp16_dir, dname), "r") as reader:
            for line in reader:
                line_data = json.loads(line)
                solution_count += 1
                ex = line_data["problem"]
                cur_steps = "### Problem: " + ex + "\n ### Solution: "
                for step in line_data["solution_sample"].split('\n'):
                    if 'answer is:' in step:
                        break
                    
                    cur_steps = cur_steps + "\n" + step
                    line_step_data = copy.deepcopy(line_data)
                    line_step_data["step_solution"] = cur_steps + '\n'
                    line_step_data['step_idx'] = steps_count
                    # group by question, solution
                    line_step_data['solution_idx'] = solution_count
                    writer.write(json.dumps(line_step_data) + "\n")
                    steps_count += 1
                    if steps_count % 10000 == 0:
                        print('-----steps count----', steps_count)
        
        if read_filecount == 2:
            read_filecount = 0
            writer.close()
            write_count += 1
            writer = open(os.path.join(steps_path_dir, str(write_count) + ".json"), "w")
            
    print('---done, steps count----', steps_count, solution_count)
    
    return True, 'OK'
    
    
if __name__ == '__main__':
    # for math_shepherd_data input
    construct_steps_data()
