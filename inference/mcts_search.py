import math
import time
import random
import scipy
from collections import defaultdict
import numpy as np
import torch as th
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

model_path = "../scripts/tmp/prm_sftgen_1e-5/checkpoint-9554/"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
gen_llm = LLM(model=model_path, trust_remote_code=True, max_model_len=8192, \
    tensor_parallel_size=1, enable_prefix_caching=True, enable_chunked_prefill=False)

# Create a class to do batch inference. Create a sampling params object.
stop_words = ["</s>", "<|im_end|>", "<|endoftext|>", "### Problem: ", "\n"]
stop_token_ids = [151645, 151643]
child_count = 4
sampling_params = SamplingParams(temperature=0.6, n=child_count, max_tokens=1024, \
    stop_token_ids=stop_token_ids, stop=stop_words)

tokenizer.padding_side = "left"
reward_llm = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation = "flash_attention_2", \
        device_map="auto", torch_dtype=th.float16)
reward_llm.eval()

# 示例：状态定义
class GameState:
    def __init__(self, path):
        self.path = path  # 游戏状态，例如棋盘状态
        self.last_action = None  # 上一步的动作
        self.child_count = child_count

    def get_possible_actions(self):
        """返回所有可能的行动"""
        outputs = self.gen_llm.generate(self.path, self.sampling_params).output[0]
        return [item.text for item in outputs]

    def apply_action(self, action):
        """根据当前行动生成新状态"""
        new_state = GameState(self.path + action)
        new_state.last_action = action
        return new_state

    def is_terminal(self):
        """检查是否为终止状态"""
        return "the answer is: " in self.path
    
    def get_reward(self):
        """获取当前状态的奖励"""        
        with th.no_grad():
            inputs_repr = self.tokenizer(self.path, \
                padding=True, truncation=True, return_tensors="pt", max_length=1024)
            inputs_repr = {k: v.to(self.device) for k, v in inputs_repr.items()}
            token_reward_logits = self.llm(**inputs_repr).logits.detach().cpu().numpy()
        
        endofstep_idx = self.tokenizer.convert_tokens
        _to_ids("<END_STEP>")
        postep_idx = self.tokenizer.convert_tokens_to_ids("<POS_STEP>")
        negstep_idx = self.tokenizer.convert_tokens_to_ids("<NEG_STEP>")
        endofstep_ids = np.where(inputs_repr["input_ids"][0].detach().cpu().numpy() == endofstep_idx)[0]
        
        # TODO: softmax on postep_idx and negstep_id, using the probablity as min input
        pos_logits = token_reward_logits[0, endofstep_ids, postep_idx]
        neg_logits = token_reward_logits[0, endofstep_ids, negstep_idx]
        postep_score = scipy.special.softmax(np.stack([pos_logits, neg_logits], axis=1), axis=1)

        return postep_score


# 定义树的节点
class Node:
    def __init__(self, state, parent=None):
        self.state = state  # 当前节点的状态
        self.parent = parent  # 父节点
        self.children = []  # 子节点
        self.visits = 0  # 该节点被访问的次数
        self.value = 0  # 节点的累积值（可能是某种评估的得分）

    def is_fully_expanded(self):
        """检查是否已扩展所有可能的子节点"""
        # 假设state提供一个可以得到所有可能动作的函数
        return len(self.children) == len(self.state.get_possible_actions())

    def best_child(self, exploration_weight=1.):
        """选择UCT算法中最好的子节点"""
        best_value = -float('inf')
        best_node = None
        for child in self.children:
            uct_value = child.value / (child.visits + 1) + exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1))
            if uct_value > best_value:
                best_value = uct_value
                best_node = child
        return best_node


# 定义蒙特卡洛树搜索类
class MCTS:
    def __init__(self, initial_state, time_limit=1.0):
        self.root = Node(initial_state)  # 初始根节点
        self.time_limit = time_limit  # 搜索时间限制

    def search(self):
        """执行MCTS搜索"""
        start_time = time.time()

        while time.time() - start_time < self.time_limit:
            node = self.select(self.root)
            reward = self.simulate(node)
            self.backpropagate(node, reward)

        return self.best_action()

    def select(self, node):
        """选择策略：按UCT值选择子节点"""
        while not node.state.is_terminal() and node.is_fully_expanded():
            node = node.best_child()
        return node

    def expand(self, node):
        """扩展节点，生成一个新的子节点"""
        actions = node.state.get_possible_actions()
        for action in actions:
            child_state = node.state.apply_action(action)
            child_node = Node(child_state, parent=node)
            node.children.append(child_node)

    def simulate(self, node):
        """模拟阶段，返回模拟的结果"""
        current_state = node.state
        while not current_state.is_terminal():
            possible_actions = current_state.get_possible_actions()
            action = random.choice(possible_actions)
            current_state = current_state.apply_action(action)
        return current_state.get_reward()

    def backpropagate(self, node, reward):
        """回溯阶段，更新节点值"""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def best_action(self):
        """选择最好的行动"""
        best_value = -float('inf')
        best_action = None
        for child in self.root.children:
            if child.visits > best_value:
                best_value = child.visits
                best_action = child.state.last_action
        return best_action



# 测试代码
if __name__ == "__main__":
    initial_state = GameState(0)
    mcts = MCTS(initial_state, time_limit=1.0)
    best_action = mcts.search()
    print(f"Best Action: {best_action}")
