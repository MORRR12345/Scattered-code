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
        return self.fc3(x)  # 返回第三层全连接层的输出

# 定义训练DQN的函数，输入参数为训练的轮数，默认为1000轮
def train_dqn():
    env = GomokuEnv()  # 创建棋盘环境
    model = DQN()  # 创建DQN模型
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器，学习率为0.001
    criterion = nn.MSELoss()  # 使用均方误差损失函数

    # 初始化胜利计数器
    win_log = np.zeros((3, algorithm.episodes))
    num_win = np.zeros((3, 1))

    # 开始训练
    for episode in tqdm(range(algorithm.episodes)):
        state = env.reset()  # 重置棋盘环境
        # 将状态转换为张量，并添加一个维度
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
        done = False  # 初始化游戏结束标志为False
        player = 1  # 机器人先手
        # 在一局游戏结束前
        while not done:
            # 机器人先手
            if player == 1:
                robot_play(state)
                next_state, win_id, done, _ = env.step(robot_play(state), player)
            # AI下棋
            else:
                state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
                # 使用模型进行前向传播，获取Q值
                q_values = model(state_tensor)
                # 可以下棋的位置
                zero_positions = torch.nonzero(state_tensor == 0)
                q_values_at_zero = q_values[0, zero_positions[:, 1]]
                max_q_value_position = torch.argmax(q_values_at_zero).item()
                # 选择Q值最大的动作
                action = np.unravel_index(zero_positions[max_q_value_position][1].item(), (board.size_x, board.size_y))
                # 执行动作，获取下一个状态、奖励、游戏是否结束和其他信息
                next_state, win_id, done, _ = env.step(action, player)
                if win_id == 2: # AI胜利
                    reward = +10 
                elif win_id == 1: # 机器人胜利
                    reward = -10 
                else: # 无事发生
                    reward = 0
                # 将下一个状态转换为张量，并添加一个维度
                next_state_tensor = torch.FloatTensor(next_state.flatten()).unsqueeze(0)
                # 计算目标Q值
                target = reward + 0.99 * torch.max(model(next_state_tensor)).item() * (1 - done)
                # 创建一个Q值的副本
                target_f = q_values.clone()
                # 更新目标Q值
                target_f[0, action[0] * board.size_x + action[1]] = target

                # 计算损失
                loss = criterion(q_values, target_f)
                optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
            state = next_state  # 更新状态
            player = 3 - player  # 换玩家
        # 记录结果
        num_win[win_id, 0] += 1
        win_log[:, episode] = num_win[:, 0]/(episode+1)
    # 保存训练好的模型参数
    torch.save(model.state_dict(), 'dqn.pth')
    # 绘制折线图
    times = list(range(algorithm.episodes))
    plt.plot(times, win_log[1], label='Robot Wins')
    plt.plot(times, win_log[2], label='AI Wins')
    plt.plot(times, win_log[0], label='Draw')
    plt.xlabel('训练次数')
    plt.ylabel('胜率')
    plt.legend()
    plt.savefig('training_plot.png')
    plt.show()

# 机器人下棋
def robot_play(state):
    zero_indices = np.where(state == 0)
    # 随机选择一个0值的位置
    random_index = np.random.choice(len(zero_indices[0]))
    # 获取随机选择的0值的位置
    action = (zero_indices[0][random_index], zero_indices[1][random_index])
    return action

# AI下棋
def dqn_play(state):
    model = DQN()  # 创建DQN模型
    model.load_state_dict(torch.load('dqn.pth', map_location=torch.device('cpu'), weights_only=True))  # 加载训练好的模型参数
    # model.load_state_dict(torch.load('dqn.pth'), weights_only=True)  # 加载训练好的模型参数
    # 将状态转换为张量，并添加一个维度
    state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
    # 使用模型进行前向传播，获取Q值
    q_values = model(state_tensor)
    # 可以下棋的位置
    zero_positions = torch.nonzero(state_tensor == 0)
    q_values_at_zero = q_values[0, zero_positions[:, 1]]
    max_q_value_position = torch.argmax(q_values_at_zero).item()
    # 选择Q值最大的动作
    action = np.unravel_index(zero_positions[max_q_value_position][1].item(), (board.size_x, board.size_y))

    return action

if __name__ == "__main__":
    train_dqn()