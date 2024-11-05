import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import map, algorithm
from TAP_env import TAPenv
from dqn import DQNAgent

# 定义训练DQN的函数
def train_dqn(i, random_rate):
    env = TAPenv()  # 创建棋盘环境
    return_count = np.zeros((algorithm.episodes)) # 初始化总回报记录器

    # 创建机器人(input_size, output_size, memory_size, random_rate)
    agent = DQNAgent((map.num_task+1)*3, map.num_task+1, 1000, random_rate)
    if i >= 1:
        agent.read_model()  # 读取模型参数

    print(f'The {i+1}/{algorithm.train_course} course:')
    ## 循环
    # for episode in range(algorithm.episodes):
    for episode in tqdm(range(algorithm.episodes)):
        state = env.reset()  # 重置地图环境
        done = 0
        # 在一次巡逻结束前
        while done == 0:
            action = agent.choose_action(state)
            next_state,reward,done = env.step(state, action)
            # 记录结果
            agent.store_transition(state, action, reward, done, next_state)
            agent.learn()
            return_count[episode] += reward
            # test(state, action, reward, done, next_state, episode)
            state = next_state

    # 保存训练好的模型参数
    agent.save_model()

    # 绘制折线图
    times = list(range(algorithm.episodes))
    plt.plot(times, return_count, label=f'The {i+1}th training session')
    plt.xlabel('time')
    plt.ylabel('return')
    plt.legend()
    plt.savefig('data/training_plot.png')

# 测试函数
def test(state, action, reward, done, next_state, episode):
    print("回合", episode, "行动", action, "奖励", round(reward, 2))
    print("状态", next_state[0, :])

if __name__ == "__main__":

    for i in range(algorithm.train_course):

        random_rate = algorithm.random_rate

        # 多次课程下
        if i >= 1:
            random_rate = algorithm.random_rate * (1 - i/(algorithm.train_course-1))

        train_dqn(i, random_rate)
    plt.show()