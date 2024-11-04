import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from gomoku_env import GomokuEnv
from config import board, algorithm

# 定义DQN类，继承自nn.Module
class DQN(nn.Module):
    # 初始化函数，输入参数为棋盘大小
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(board.size_x * board.size_y, algorithm.linear[0])  # 定义第一层全连接层
        self.fc2 = nn.Linear(algorithm.linear[0], algorithm.linear[1])  # 定义第二层全连接层
        self.fc3 = nn.Linear(algorithm.linear[1], board.size_x * board.size_y)  # 定义第三层全连接层

    # 前向传播函数
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 第一层全连接层的激活函数为ReLU
        x = torch.relu(self.fc2(x))  # 第二层全连接层的激活函数为ReLU
        x = self.fc3(x)
        return x # 返回第三层全连接层的输出

# DQN智能体
class DQNAgent:
    def __init__(self, input_size, output_size, memory_size, random_rate):
        # 初始化记忆库，用于存储经验
        self.memory = np.zeros((memory_size, input_size + output_size + 3))
        # 记忆库计数器，用于记录当前记忆库中的经验数量
        self.memory_counter = 0
        # 学习步骤计数器，用于记录模型已经学习的步骤数量
        self.learn_step_counter = 0
        # 记忆库大小
        self.memory_size = memory_size
        # 输入输出大小
        self.input_size = input_size
        self.output_size = output_size
        # 随机行动概率
        self.random_rate = random_rate
        # 创建DQN模型
        self.dqn = DQN()
        # 创建优化器，使用Adam算法，学习率为0.001
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        # 创建损失函数，使用均方误差损失
        self.loss_func = nn.MSELoss()

    # 选择行动
    def choose_action(self, x, player):
        if player == 1:
            state = np.where(x != 0, 3-x, x)
        else:
            state = x
        # 将输入转换为张量
        # state = torch.unsqueeze(torch.FloatTensor(x), 0)
        state = torch.FloatTensor(state.flatten()).unsqueeze(0)
        # 可以下棋的位置
        zero_pos = torch.nonzero(state == 0)

        # 如果随机数大于随机概率，选择动作值最大的动作，否则随机选择一个动作
        if np.random.uniform() > self.random_rate:
            # 使用DQN模型预测动作值
            action_value = self.dqn.forward(state)
            # 可以下棋地方的价值
            action_values_at_zero = action_value[0, zero_pos[:, 1]]
            # 选取最大价值位置作为行动
            action_pos = torch.argmax(action_values_at_zero).item()
        else:
            # 随机选取位置
            action_pos = torch.randint(0, len(zero_pos), (1,)).item()
        action = np.unravel_index(zero_pos[action_pos][1].item(), (board.size_x, board.size_y))
        return action

    # 存储训练数据
    def store_transition(self, s, a, r, s_):
        # 将当前状态、动作、奖励和下一个状态打包成一个经验
        transition = np.hstack((s, a, r, s_))
        # 计算存储位置，如果记忆库已满，则覆盖最旧的经验
        index = self.memory_counter % self.memory_size
        # 将经验存储到记忆库中
        self.memory[index, :] = transition
        # 更新记忆库计数器
        self.memory_counter += 1

    # 训练模型
    def learn(self):
        # 每50步更新一次目标DQN模型
        if self.learn_step_counter % 50 == 0:
            self.target_dqn = DQN()
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        # 如果记忆库已满，随机选择128个经验，否则随机选择当前记忆库中的经验
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=128)
        else:
            sample_index = np.random.choice(self.memory_counter, size=128)
        # 从记忆库中获取经验
        batch_memory = self.memory[sample_index, :]
        # 将经验分解为当前状态、动作、奖励和下一个状态
        b_s = torch.FloatTensor(batch_memory[:, :self.input_size])
        b_a = torch.FloatTensor(batch_memory[:, self.input_size:(self.input_size+self.output_size)].astype(int))
        b_r = torch.FloatTensor(batch_memory[:, (self.input_size+self.output_size):(self.input_size+self.output_size+1)])
        b_s_ = torch.FloatTensor(batch_memory[:, -self.input_size:])
        # 使用DQN模型预测当前状态的动作值
        q_eval = self.dqn(b_s).gather(1, b_a.int())
        # 使用目标DQN模型预测下一个状态的最大动作值
        q_next = self.target_dqn(b_s_).detach()
        # 计算目标Q值
        q_target = b_r + 0.9 * q_next.max(1)[0].view(128, 1)
        # 计算损失
        loss = self.loss_func(q_eval, q_target)
        # 清空梯度
        self.optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 更新模型参数
        self.optimizer.step()
        # 更新学习步骤计数器
        self.learn_step_counter += 1

    # 保存模型信息
    def save_model(self):
        torch.save(self.dqn.state_dict(), 'dqn.pth')
    
    # 读取模型信息
    def read_model(self):
        self.dqn.load_state_dict(torch.load('dqn.pth', map_location=torch.device('cpu')))  # 加载训练好的模型参数 , weights_only=True

# 定义训练DQN的函数
def train_dqn(i):
    env = GomokuEnv()  # 创建棋盘环境
    # 创建机器人(input_size, output_size, memory_size, random_rate)
    if i == 0:
        agent_1 = DQNAgent(board.size_x*board.size_y, board.size_x*board.size_y, 1000, 1.0) # 随机机器人
    else:
        agent_1 = DQNAgent(board.size_x*board.size_y, board.size_x*board.size_y, 1000, 0.0) # AI机器人
        agent_1.read_model() 
    # 要训练的机器人
    agent_2 = DQNAgent(board.size_x*board.size_y, board.size_x*board.size_y, 1000, 0.2)

    # 初始化胜利计数器
    win_count = np.zeros((3, 1))
    win_rate = np.zeros((3, algorithm.episodes))

    # 开始训练
    # for episode in range(algorithm.episodes):
    for episode in tqdm(range(algorithm.episodes)):
        state = env.reset()  # 重置棋盘环境
        done = False  # 初始化游戏结束标志为False
        player = 1
        reward = 0
        # 在一局游戏结束前
        while not done:
            if player == 1:
                # 先手机器人
                action_1 = agent_1.choose_action(state, player)
                next_state, win_id, done, _ = env.step(action_1, player)
                state_1 = next_state.flatten()
            else:
                state_2 = state.flatten()
                # 后手机器人(要训练的AI)
                action_2 = agent_2.choose_action(state, player)
                next_state, win_id, done, _ = env.step(action_2, player)

            if done:
                if win_id == 2:
                    reward = +2
                elif win_id == 1:
                    reward = -2
                else:
                    reward = -1
            else:
                reward = 0

            if player == 2:
                agent_2.store_transition(state_2, action_2, reward, state_1)

            state = next_state  # 更新状态
            player = 3 - player # 更换选手
        agent_2.learn()
        # 记录结果
        win_count[win_id, 0] += 1
        win_rate[:, episode] = win_count[:, 0]/(episode+1)

    # 保存训练好的模型参数
    agent_2.save_model()

    # 绘制折线图
    times = list(range(algorithm.episodes))
    plt.plot(times, win_rate[0], label='Draw')
    plt.plot(times, win_rate[1], label='Robot Wins')
    plt.plot(times, win_rate[2], label='AI Wins')
    plt.xlabel('训练次数')
    plt.ylabel('胜率')
    plt.legend()
    plt.savefig('training_plot.png')
    plt.show()

# AI下棋
def robot_play(state, ID):
    agent = DQNAgent(board.size_x*board.size_y, board.size_x*board.size_y, 1000, 0.0) # AI机器人
    agent.read_model()
    if ID == 1: #AI身份转为机器人（对手）
        state_re = np.where(state != 0, 3-state, state)
    action = agent.choose_action(state_re)
    return action

if __name__ == "__main__":

    for i in range(algorithm.train_course):
        train_dqn(i)