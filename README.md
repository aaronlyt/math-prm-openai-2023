# math-prm-openai-2023
reproduce Let’s Verify Step by Step and MATH-SHEPHERD data construction process for learning 

复现文章: Let’s Verify Step by Step
其它参考文章:
Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
MATH-SHEPHERD: VERIFY AND REINFORCE LLMS STEP-BY-STEP WITHOUT HUMAN ANNOTATIONS

notes: 可能有些路径需要自己设置下；同时主要用于学习，可读性一般吧，感兴趣参考下

主要实验内容:
1. 基于grade math数据集，训练verify model
2. 参考Let’s Verify Step by Step，训练reward model
- 使用原始数据集
- 参考MATH-SHEPHERD构建prm reward数据集，训练reward model


一. verify model 实验

1. 框架: trl, transformers 
2. 基模: qwen2.5 1.5b BASE
3. 训练数据: grade school math训练数据集
4. 运行流程
- base model sft 2 epoch: matho1/scripts/sft_gsm.py
- sft model为verify model训练构建采样数据(多vllm实例推理): matho1/inference/grade_math/sample_4verifymodel_.py
- 训练verify model: matho1/scripts/verify_gsm_trainer.py or matho1/scripts/verify_gsm.py
- 使用verify model排序sft model的采样数据: matho1/inference/grade_math/verify_rank.py
- 评测脚本: matho1/inference/grade_math/eval_ranker.py
5. 评测结果: grade school math test数据集, accuracy: 0.4147

二.  参考Let’s Verify Step by Step, 训练reward model

1. 框架: trl, transformers 
2. 基模: qwen2.5 1.5b BASE
3. 训练数据: prm800k数据集
4. prm800k运行流程
- prm800k数据预处理: matho1/inference/prm800k/prm_dataset.py： prod Defaults to False. 有多条路径只选择一条，True多个选项会进行组合
- few shot方式产生generator sft训练数据集: matho1/inference/prm800k/fewshot_4gensft.py
- generator sft: matho1/scripts/sft_prm.py
- 使用训练的generator采样生成prm reward训练数据: matho1/inference/prm800k/gen_mathslu.py，同时也可以用来采样测试数据集
- prm reward model训练: matho1/scripts/nexttok_prm_trainer.py, PRMs are trained for 2 epochs. On smaller datasets (such as in phase 1 and the first few generations of phase 2) this improves the final performance over training for just 1 epoch
- using process reward model to rank the generator sampling on test dataset: matho1/scripts/prm_rank.py
- 统计评测: matho1/scripts/gpstate_ranker_score
评测结果: 没有majority voting效果好
- 最好的一次实验结果: accuracy: 0.46(最优结果是prod=True)
- sft generator majority voting: accuracy: 0.508

分析: 可能数据分布不一致问题，参考Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters：We found that it was easy to exploit a PRM trained on this dataset via even naïve
strategies such as best-of-N sampling. We hypothesize that this is likely a result of the distribution shift between the GPT-4 generated samples in their dataset and our PaLM 2 models

next: 所以按照Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters方式构建MATH-SHEPHERD数据集

三  MATH-SHEPHERD数据构建流程
- prm800k数据预处理: matho1/inference/prm800k/prm_dataset.py
- few shot方式产生generator sft训练数据集: matho1/inference/prm800k/fewshot_4gensft.py
- generator sft: matho1/scripts/sft_prm.py------这里也不一定要微调，没空做对比实验了
- 使用训练的generator采样生成prm reward训练数据 matho1/inference/prm800k/gen_mathslu.py，同时也可以用来采样测试数据集
- 为了高效推理组织成steps格式数据: matho1/inference/prm800k/shepsherd_steps.py
- MATH-SHEPHERD核心代码-采样构建reward信号: matho1/inference/prm800k/math_shepherd_data.py
- 数据后处理: matho1/inference/prm800k/math_shepherd_agg.py, aggregate the math-shepherd sampling dataset, execute after math_shepherd_data
- prm reward model训练: matho1/scripts/nexttok_prm_trainer.py
- using process reward model to rank the generator sampling on test dataset: matho1/scripts/prm_rank.py
- 统计评测: matho1/scripts/gpstate_ranker_score
评测结果: accuracy: 0.418

存在的问题: reward信号构建过程，无法采样足够的数据集，猜测可能是sft过拟合了

后续可以做的: 
- 要么Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters文章中使用base model + few shot方式产生
- 要么MATH-SHEPHERD文章中做法
