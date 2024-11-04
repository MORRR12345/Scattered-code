import numpy as np
import random 
import copy

#-----------------------智能体-----------------------
class Agent():

    # 初始化
    def __init__(self,ID,epsilon,learn_rate):
        self.id = ID 
        self.epsilon = epsilon #随机率
        self.learn_rate = learn_rate #学习速率
        self.value = dict()  #状态价值字典
        self.reward = 0
        self.state_before = np.zeros(9) #之前状态

    # 下棋
    def play(self,state):
        decimal = np.random.uniform(0,1)
        if decimal <= self.epsilon: #随机下棋
            available = np.where(state==0)
            length = len(available[0])
            if length > 0: #如果没有下完
                choose = np.random.randint(length)
                state[available[0][choose],available[1][choose]] = self.id
                return state
        else: #抉择最大价值状态下棋
            long_state = state.flatten()
            max_value = 0
            max_state = []
            for i in range(len(long_state)):
                if long_state[i] == 0: #出现空位
                    long_state[i] = self.id
                    self.check_value(long_state)
                    val = self.value.get(tuple(long_state))
                    if val > max_value:
                        max_value = val
                        max_state = []
                        max_state.append(copy.deepcopy(long_state))
                    elif val == max_value:
                        max_state.append(copy.deepcopy(long_state))
                    long_state[i] = 0
            long_state = random.choice(max_state)
            long_state = long_state.reshape((3, 3))
            return long_state

    # 更新状态价值
    def update_value(self,state,win_ID):
        long_state = state.flatten()
        if win_ID == 0:
            reward = 0
        elif win_ID == self.id:
            reward = 1
        else:
            reward = -1
        self.check_value(long_state)
        self.check_value(self.state_before)
        val = self.value.get(tuple(self.state_before))
        val = val + self.learn_rate*(reward + self.value.get(tuple(long_state)) - val)
        self.value[tuple(self.state_before)] = val
        self.state_before = long_state

    # 检查是否在字典中
    def check_value(self,long_state):
        if self.value.get(tuple(long_state)) == None:
            self.value[tuple(long_state)] = 3

    # 下一状态价值
    def next_value(self,long_state):
        sum_value = 0
        for i in range(len(long_state)):
            if long_state[i] == 0:
                long_state[i] = self.id
                sum_value += self.value.get(tuple(long_state)) 
                long_state[i] = self.id
        return sum_value
    
    # 重置
    def reset(self):
        self.state_before  = np.zeros(9)