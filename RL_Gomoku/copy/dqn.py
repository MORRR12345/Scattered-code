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
    
# 只在可以下棋的地方下棋
def model_only(model, state):
    # 使用模型进行前向传播，获取Q值
    q_values = model(state)
    # 可以下棋的位置
    zero_positions = torch.nonzero(state == 0)
    q_values_at_zero = q_values[0, zero_positions[:, 1]]
    q_max,_ = torch.max(q_values_at_zero, 0)
    max_q_value_position = torch.argmax(q_values_at_zero).item()
    action = np.unravel_index(zero_positions[max_q_value_position][1].item(), (board.size_x, board.size_y))
    return q_max, q_values, action

# DQN智能体
class DQNAgent:
    def __init__(self, input_size, hidden_size, output_size, memory_size):
        # 初始化记忆库，用于存储经验
        self.memory = np.zeros((memory_size, input_size + output_size + 1))
        # 记忆库计数器，用于记录当前记忆库中的经验数量
        self.memory_counter = 0
        # 学习步骤计数器，用于记录模型已经学习的步骤数量
        self.learn_step_counter = 0
        # 记忆库大小
        self.memory_size = memory_size
        # 创建DQN模型
        self.dqn = DQN(input_size, hidden_size, output_size)
        # 创建优化器，使用Adam算法，学习率为0.001
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        # 创建损失函数，使用均方误差损失
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # 将输入转换为张量
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # 使用DQN模型预测动作值
        action_value = self.dqn.forward(x)
        # 如果随机数大于0.2，选择动作值最大的动作，否则随机选择一个动作
        if np.random.uniform() > 0.2:
            action = torch.max(action_value, 1)[1].data.numpy()
        else:
            action = np.random.randint(0, 2)
        return action

    def store_transition(self, s, a, r, s_):
        # 将当前状态、动作、奖励和下一个状态打包成一个经验
        transition = np.hstack((s, [a, r], s_))
        # 计算存储位置，如果记忆库已满，则覆盖最旧的经验
        index = self.memory_counter % self.memory_size
        # 将经验存储到记忆库中
        self.memory[index, :] = transition
        # 更新记忆库计数器
        self.memory_counter += 1

    def learn(self):
        # 每50步更新一次目标DQN模型
        if self.learn_step_counter % 50 == 0:
            self.target_dqn = DQN(4, 16, 2)
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        # 如果记忆库已满，随机选择128个经验，否则随机选择当前记忆库中的经验
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=128)
        else:
            sample_index = np.random.choice(self.memory_counter, size=128)
        # 从记忆库中获取经验
        batch_memory = self.memory[sample_index, :]
        # 将经验分解为当前状态、动作、奖励和下一个状态
        b_s = torch.FloatTensor(batch_memory[:, :4])
        b_a = torch.LongTensor(batch_memory[:, 4:5].astype(int))
        b_r = torch.FloatTensor(batch_memory[:, 5:6])
        b_s_ = torch.FloatTensor(batch_memory[:, -4:])
        # 使用DQN模型预测当前状态的动作值
        q_eval = self.dqn(b_s).gather(1, b_a)
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

# 定义训练DQN的函数
def train_dqn(adversary):
    env = GomokuEnv()  # 创建棋盘环境
    model = DQN()  # 创建DQN模型
    if adversary == "robot":
        # 机器人作为对手
        adversary_play = robot_play
        print(adversary)
    else:
        #曾经的自己作为对手
        adversary_play = ai_play
        print(adversary)
        model.load_state_dict(torch.load('dqn.pth', map_location=torch.device('cpu')))  # 加载训练好的模型参数 , weights_only=True
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器，学习率为0.001
    criterion = nn.MSELoss()  # 使用均方误差损失函数

    # 初始化胜利计数器
    win_count = np.zeros((3, 1))
    win_rate = np.zeros((3, algorithm.episodes))

    # 开始训练
    # for episode in range(algorithm.episodes):
    for episode in tqdm(range(algorithm.episodes)):
        state = env.reset()  # 重置棋盘环境
        done = False  # 初始化游戏结束标志为False
        player = 1  # 机器人先手
        robot_ID = 1 # 机器人先手
        AI_ID = 2 #AI默认2是自己
        state, win_id, done, _ = env.step(adversary_play(state, player), player)
        # 在一局游戏结束前
        while not done:

            # AI下棋
            player = AI_ID # 2
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
            # 获取Q值, 并在可下区域抉择行动
            _, q_values, action_AI = model_only(model, state_tensor)
            next_state, win_id, done, _ = env.step(action_AI, AI_ID)
            # print("AI: ", action_AI)
            # print(next_state)
            if done :
                if win_id == 2:
                    reward = +2 #AI胜
                elif win_id == 0:
                    reward = -1 #平局
                target = reward
            else: #无事发生

                # 机器人下棋
                player = robot_ID # 1
                action_robot = adversary_play(next_state, player)
                next_state, win_id, done, _ = env.step(action_robot, robot_ID)
                # print("robot: ", action_robot)
                # print(next_state)
                if done :
                    if win_id == 1:
                        reward = -2 #AI输
                    elif win_id == 0:
                        reward = -1 #平局
                    target = reward
                else: 
                    reward = 0 #无事发生
                    # 将下一个状态转换为张量，并添加一个维度
                    next_state_tensor = torch.FloatTensor(next_state.flatten()).unsqueeze(0)
                    q_max,_,_ = model_only(model, next_state_tensor)
                    target = reward + 0.99 * q_max.item() * (1-done)

            # 创建一个Q值的副本
            target_f = q_values.clone()
            # 更新目标Q值
            target_f[0, action_AI[0] * board.size_x + action_AI[1]] = target

            # 计算损失
            loss = criterion(q_values, target_f)
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            state = next_state  # 更新状态
            # time.sleep(1) #休息1秒
        # 记录结果
        win_count[win_id, 0] += 1
        win_rate[:, episode] = win_count[:, 0]/(episode+1)
        # print(state)
        # print(win_count)
        # time.sleep(1) #休息1秒
    # 保存训练好的模型参数
    torch.save(model.state_dict(), 'dqn.pth')
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

# 机器人下棋
def robot_play(state,ID):
    zero_indices = np.where(state == 0)
    # 随机选择一个0值的位置
    random_index = np.random.choice(len(zero_indices[0]))
    # 获取随机选择的0值的位置
    action = (zero_indices[0][random_index], zero_indices[1][random_index])
    return action

# AI下棋
def ai_play(state,ID):
    model2 = DQN()  # 创建DQN模型
    model2.load_state_dict(torch.load('dqn.pth', map_location=torch.device('cpu')))  # 加载训练好的模型参数 , weights_only=True
    # 将状态转换为张量，并添加一个维度
    state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
    if ID == 1: #AI身份转为机器人（对手）
        state_tensor = torch.where(state_tensor != 0, 3 - state_tensor, state_tensor)
    _,_,action = model_only(model2, state_tensor)
    return action

if __name__ == "__main__":

    for i in range(algorithm.train_course):
        if i == 1:
            adversary = "robot"
        else: 
            adversary = "ai"
        train_dqn(adversary)