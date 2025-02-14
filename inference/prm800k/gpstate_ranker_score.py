import os
import sys
import json
import glob
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from grader import grade_answer


if __name__ == "__main__":
    base_dir = "../../datasets_repo/prm800k/prm800k/"
    ranker_dir = os.path.join(base_dir, "sampling_data", "shepsherd_8e-05_ep10")
    
    ranker_datas = []
    for dname in glob.glob(os.path.join(ranker_dir, "*.json")):
        with open(os.path.join(ranker_dir, dname), "r") as reader:
            for line in reader:
                line_data = json.loads(line)
                try:
                    pred_answer = None
                    for cand_ansline in line_data['solution_sample'].split('\n'):
                        if 'The answer is:' in cand_ansline:
                            pred_answer = cand_ansline.split("The answer is: ")[1].strip()
                            break
                    if not pred_answer:
                        print('-------no pred answer---', line_data['solution_sample'], line_data['reward_logits_sum'])
                except:
                    print('-pred answer no answer--', line_data['solution_sample'], line_data['reward_logits_sum'])
                    pred_answer = None
                batch_label = int(grade_answer(pred_answer, line_data['answer']))
                ranker_datas.append([line_data['problem'], line_data['answer'], \
                    pred_answer, line_data['reward_logits_sum'], batch_label])
    
    print('-----data length---', len(ranker_datas))
    ranker_dataframe = pd.DataFrame(ranker_datas, columns=["question", "raw_answer", "pred_answer", "reward_logits_sum", "label"])
    
    ranker_dataframe_gp = ranker_dataframe.loc[ranker_dataframe.groupby("question")["reward_logits_sum"].idxmax()]
    corr_aftranker = ranker_dataframe_gp[ranker_dataframe_gp['label'] == 1]
    print('---correct count after ranker and total count---', corr_aftranker.shape[0], ranker_dataframe_gp.shape[0])
    print('---accuracy after ranker---', corr_aftranker.shape[0] / ranker_dataframe_gp.shape[0])
    
    # double check
    corr_count = 0
    for key, gp_data in ranker_dataframe.groupby("question"):
        max_index = gp_data['reward_logits_sum'].idxmax()
        if gp_data.loc[max_index]['label'] == 1:
            corr_count += 1
    print('---doublecheck correct count after ranker and total count---', corr_count, ranker_dataframe.shape[0])
    
    # see top-N accuracy    
    # 根据 question 分组，对每组中的 label 列应用 max() 操作。由于 label 的取值是 0 或 1，如果存在 1，结果就是 1；否则是 0
    group_result = ranker_dataframe.groupby("question")["label"].max().reset_index()
    # 计算正确率
    total_questions = group_result.shape[0]
    correct_questions = group_result["label"].sum()
    accuracy = correct_questions / total_questions
    print("raw sft topcorr_bfranker",  accuracy, group_result.shape[0])
    
    # see majority vote accuracy
    # 统计每组中 pred_answer 的出现次数
    sampling_most_common = (
        ranker_dataframe.groupby(["question", "pred_answer"])
        .size()
        .reset_index(name="count")  # 添加计数列
    )
    # 对每个 question 选择 count 最大的条目
    majority_result = sampling_most_common.loc[sampling_most_common.groupby("question")["count"].idxmax()]
    
    # 如果需要将结果与原数据合并：
    majority_merge_result = pd.merge(majority_result, ranker_dataframe, on=["question", "pred_answer"], how='left')
    majority_merge_result = majority_merge_result.drop_duplicates(subset=['question'])
    majority_corr_result = majority_merge_result[majority_merge_result['label'] == 1]
    print('---majority accuracy---',majority_corr_result.shape[0], majority_merge_result.shape[0], majority_corr_result.shape[0] / majority_merge_result.shape[0])

    # double check
    mjcorr_count = 0
    for _, gp_data in ranker_dataframe.groupby("question"):
        max_count = 0
        max_idxlabel  = -1
        for _, subrow in gp_data.groupby('pred_answer'):
            if subrow.shape[0] > max_count:
                max_count = subrow.shape[0]
                max_idxlabel = subrow.iloc[0]['label']
        
        if max_idxlabel == 1:
            mjcorr_count += 1
    print('---doublecheck majority accuracy---', mjcorr_count)

""" 
step token 0-1 prediction, one-stage train on large only
-----data length--- 16755
---correct count after ranker and total count--- 180 500
---accuracy after ranker--- 0.36
---correct count after ranker and total count--- 180 16755
raw sft topcorr_bfranker 0.802 500
---majority accuracy--- 254 500 0.508
---majority accuracy--- 254

# large prod 1 epoch
-----data length--- 16755
---correct count after ranker and total count--- 188 500
---accuracy after ranker--- 0.376

# large prod 2 epoch, 8e-5, dropout 0.1,  0.20 loss, ./tmp/trainersc_prmlgeprod_nttk_8e-05
---correct count after ranker and total count--- 214 500
---accuracy after ranker--- 0.428

# large prod 2 epoch, 8e-5, dropout 0.1,  0.20 loss, inference cal softmax, trainer_prmlgeprod_nttk_softscore
---correct count after ranker and total count--- 219 500
---accuracy after ranker--- 0.438

# large prod 9 epoch, dev 0.14 loss,
---correct count after ranker and total count--- 204 500
---accuracy after ranker--- 0.408????

# large prod 5 epoch, dev 0.19 loss, 
---correct count after ranker and total count--- 208 500
---accuracy after ranker--- 0.416

# large prod 2 epoch, dev 0.25 loss, no embedding decay; woembdecay_ep2_prmlgeprod_nttk_1e-05/checkpoint-2952/
-----data length--- 16755
---correct count after ranker and total count--- 230 500
---accuracy after ranker--- 0.46

# large prod 2 epoch, large loss, eval_loss': 8.037403106689453, no embedding decay, sum loss, learning rate is not ok?
-----data length--- 16755
---correct count after ranker and total count--- 172 500
---accuracy after ranker--- 0.344
"""