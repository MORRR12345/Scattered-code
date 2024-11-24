import torch
import torch.nn as nn
import torch.optim as optim
from config import cfg

class net(nn.Module):
    # 初始化函数，输入参数为棋盘大小
    def __init__(self, input_size, output_size):
        super(net, self).__init__()
        self.fc1 = nn.Linear(input_size, cfg.linear[0])  # 定义第一层全连接层
        self.fc2 = nn.Linear(cfg.linear[0], cfg.linear[1])  # 定义第二层全连接层
        self.fc3 = nn.Linear(cfg.linear[1], output_size)  # 定义第三层全连接层

    # 前向传播函数
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 第一层全连接层的激活函数为ReLU
        x = torch.relu(self.fc2(x))  # 第二层全连接层的激活函数为ReLU
        x = self.fc3(x)
        return x # 返回第三层全连接层的输出

# 定义Deep类
class Deep():
    # 初始化函数，输入参数为棋盘大小
    def __init__(self, input_size, output_size):
        # 创建对象
        self.net = net(input_size, output_size)
        # 创建优化器，使用Adam算法，学习率为0.001
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        # 创建损失函数，使用均方误差损失
        self.loss_func = nn.MSELoss()
    
    def learn(self, x, y):
        x = torch.tensor(x, dtype=torch.float).unsqueeze(0)
        # print(x)
        target = torch.tensor(y, dtype=torch.float).unsqueeze(0)
        get = self.net.forward(x)
        # print(get)
        # 计算损失
        loss = self.loss_func(get, target)
        # 清空梯度
        self.optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 更新模型参数
        self.optimizer.step()
        return loss.item()
    
    def get_y(self, x):
        x = torch.tensor(x, dtype=torch.float).unsqueeze(0)
        y = self.net.forward(x)
        return y.item()
