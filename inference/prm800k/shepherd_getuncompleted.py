import os
import sys
import json
import glob
from typing import Any, Dict, List
import numpy as np
import pandas as pd


if __name__ == '__main__':
    
    base_dir = "../../datasets_repo/prm800k/"
    shepherd_dir = os.path.join(base_dir, "prm800k/shepsherd_data", "shepherd_annotation_steps_data")
    
    shepherd_problems = {}
    complete_sample_count = 0
    for dname in glob.glob(os.path.join(shepherd_dir, "*.json")):
        with open(os.path.join(shepherd_dir, dname), "r") as reader:
            for line in reader:
                line_data = json.loads(line)
                unique_id = line_data['unique_id']
                complete_sample_count += 1
                if unique_id not in shepherd_problems:
                    shepherd_problems[unique_id] = 1
    print('---------completed unque problem---', len(shepherd_problems), complete_sample_count)
    
    base_dir = "../../datasets_repo/prm800k/prm800k/"
    steps_path = os.path.join("../../datasets_repo/prm800k/shepsherd_data/", "mathtra16_steps.jsonl")
    uncomplete_path = os.path.join("../../datasets_repo/prm800k/shepsherd_data/", "mathtra16_steps_attempt2.jsonl")
    uncomplete_sample_count = 0
    with open(steps_path, 'r') as reader, open(uncomplete_path, 'w') as writer:
        for line in reader:
            line_data = json.loads(line)
            if line_data['unique_id'] in shepherd_problems:
                continue
            uncomplete_sample_count += 1
            writer.write('%s\n' % json.dumps(line_data))
    
    print('--uncomplete  sampling count is---', uncomplete_sample_count)

    