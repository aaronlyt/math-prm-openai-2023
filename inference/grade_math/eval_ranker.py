import os
import sys
import json
import glob
from typing import Any, Dict, List
import numpy as np
import pandas as pd


if __name__ == "__main__":
    base_dir = "../../datasets_repo/grade-school-math/grade_school_math/"
    ranker_dir = os.path.join(base_dir, "verify_data", "verify_ranker")
    
    ranker_datas = []
    for dname in glob.glob(os.path.join(ranker_dir, "*.json")):
        with open(os.path.join(ranker_dir, dname), "r") as reader:
            for line in reader:
                line_data = json.loads(line)
                ranker_datas.append([line_data['question'], line_data['raw_answer'], \
                    line_data['answer_sample'], line_data['batch_sample_label'], line_data['reward_logits_sum']])
    
    print('-----data length---', len(ranker_datas))
    ranker_dataframe = pd.DataFrame(ranker_datas, columns=["question", "raw_answer", "answer_sample", "batch_sample_label", "reward_logits_sum"])

    ranker_dataframe = ranker_dataframe.loc[ranker_dataframe.groupby("question")["reward_logits_sum"].idxmax()]
    corr_aftranker = ranker_dataframe[ranker_dataframe['batch_sample_label'] == 1]
    print('---correct count after ranker and total count---', corr_aftranker.shape[0], ranker_dataframe.shape[0])
    print('---accuracy after ranker---', corr_aftranker.shape[0] / ranker_dataframe.shape[0])
    
    """
    -----data length--- 131900
    ---correct count after ranker and total count--- 547 1319
    ---accuracy after ranker--- 0.41470811220621684
    """