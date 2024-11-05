import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import map, algorithm

# 定义DQN类，继承自nn.Module
class DQN(nn.Module):
    # 初始化函数，输入参数为棋盘大小
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, algorithm.linear[0])  # 定义第一层全连接层
        self.fc2 = nn.Linear(algorithm.linear[0], algorithm.linear[1])  # 定义第二层全连接层
        self.fc3 = nn.Linear(algorithm.linear[1], output_size)  # 定义第三层全连接层

    # 前向传播函数
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 第一层全连接层的激活函数为ReLU
        x = torch.relu(self.fc2(x))  # 第二层全连接层的激活函数为ReLU
        x = self.fc3(x)
        return x # 返回第三层全连接层的输出

# DQN智能体
class DQNAgent:
    def __init__(self, input_size, output_size, memory_size, random_rate):
        # 初始化记忆库，用于存储经验            s, a, r, d, s_
        self.memory = np.zeros((memory_size, 2*input_size +3))
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
        self.dqn = DQN(input_size, output_size)
        # 创建优化器，使用Adam算法，学习率为0.001
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        # 创建损失函数，使用均方误差损失
        self.loss_func = nn.MSELoss()

    # 选择行动
    def choose_action(self, x):
        state = x.copy()
        # 将输入转换为张量
        state = torch.FloatTensor(state.flatten()).unsqueeze(0)
        # 如果随机数大于随机概率，选择动作值最大的动作，否则随机选择一个动作
        if np.random.uniform() > self.random_rate:
            # 使用DQN模型预测动作值
            action_value = self.dqn.forward(state)
            # 选取最大价值位置作为行动
            action = torch.argmax(action_value).item()
        else:
            # 随机选取位置
            action = np.random.choice(map.num_task+1)
        return action

    # 存储训练数据
    def store_transition(self, s, a, r, d, s_):
        # 将当前状态、动作、奖励和下一个状态打包成一个经验
        transition = np.hstack((s.flatten(), a, r, d, s_.flatten()))
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
            self.target_dqn = DQN(self.input_size, self.output_size)
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        # 如果记忆库已满，随机选择128个经验，否则随机选择当前记忆库中的经验
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=128)
        else:
            sample_index = np.random.choice(self.memory_counter, size=128)
        # 从记忆库中获取经验
        batch_memory = self.memory[sample_index, :]

        # 将经验分解
        b_s = torch.FloatTensor(batch_memory[:, :self.input_size])
        b_a = torch.FloatTensor(batch_memory[:, self.input_size:(self.input_size+1)]).type(torch.int64)
        b_r = torch.FloatTensor(batch_memory[:, (self.input_size+1):(self.input_size+2)])
        b_d = torch.FloatTensor(batch_memory[:, (self.input_size+2):(self.input_size+3)])  # 取出是否结束的标志
        b_s_ = torch.FloatTensor(batch_memory[:, -self.input_size:])

        # 使用DQN模型预测当前状态的动作值
        q_eval = self.dqn(b_s).gather(1, b_a)
        # 使用目标DQN模型预测下一个状态的最大动作值
        q_next = self.target_dqn(b_s_).detach()
        # 计算目标Q值
        q_target = b_r + 0.9 * q_next.max(1)[0].view(128, 1) * (1-b_d)
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
        torch.save(self.dqn.state_dict(), 'data/dqn.pth')
    
    # 读取模型信息
    def read_model(self):
        self.dqn.load_state_dict(torch.load('data/dqn.pth', map_location=torch.device('cpu')))  # 加载训练好的模型参数