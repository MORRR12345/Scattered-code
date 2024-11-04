import numpy as np
import random 
#-----------------------智能体-----------------------
class Agent_init():
    #初始化
    def __init__(self,OOXX_index,Epsilon=0.1,LearningRate=0.1):
        self.value = np.zeros((3,3,3,3,3,3,3,3,3)) #状态价值
        self.currentState = np.zeros(9) #当前状态
        self.previousState = np.zeros(9) #之前状态
        self.index = OOXX_index 
        self.epsilon = Epsilon #随机率
        self.alpha = LearningRate #学习速率
    #重置
    def reset(self):
        self.currentState = np.zeros(9) #当前状态
        self.previousState = np.zeros(9) #之前状态
    #行动
    def actionTake(self,State):
        state = State.copy()
        available = np.where(state==0)[0]
        length = len(available)
        if length == 0: #如果下完
            return state
        else:
            random = np.random.uniform(0,1)
            if random<self.epsilon: #随机策略
                choose = np.random.randint(length)
                state[available[choose]] = self.index
            else:
                tempValue = np.zeros(length)
                for i in range(length):
                    tempState = state.copy()
                    tempState[available[i]] = self.index
                    tempValue[i] =self.value[tuple(tempState.astype(int))]
                choose = np.where(tempValue==np.max(tempValue))[0]
                chooseIndex = np.random.randint(len(choose))
                state[available[choose[chooseIndex]]] = self.index
    #值更新
    def valueUpdate(self,State):
        self.currentState = State.copy()
        self.value[tuple(self.previousState.astype)] += \
        self.alpha*(self.value[tuple(self.currentState.astype(int))])
        self.value[tuple(self.previousState.astype(int))]
        self.previousState = self.currentState.copy()

#-----------------------智能体-----------------------
class Agent_change():
    #初始化
    def __init__(self,OOXX_index,Epsilon=0.1,LearningRate=0.1):
        self.value = 3*np.ones((3,3))  #状态价值
        self.currentState = np.zeros(9) #当前状态
        self.previousState = np.zeros(9) #之前状态
        self.id = OOXX_index 
        self.epsilon = Epsilon #随机率
        self.alpha = LearningRate #学习速率
    #重置
    def reset(self):
        self.currentState = np.zeros(9) #当前状态
        self.previousState = np.zeros(9) #之前状态
    #行动
    def play(self,State):
        state = State.copy()
        available = np.where(state==0) #寻找可以落子的地方
        length = len(available[0])
        if length > 0: #如果没有下完
            random_num = np.random.uniform(0,1)
            if random_num < self.epsilon: #随机策略
                choose = np.random.randint(length)
                state[available[0][choose],available[1][choose]] = self.id
            else:
                tempValue = np.zeros(length) #学习策略
                for i in range(length):
                    tempState = state.copy()
                    tempState[available[0][i],available[1][i]] = self.id
                    tempValue[i] =self.value[tuple(tempState.astype(int))]
                choose = np.where(tempValue==np.max(tempValue))
                chooseIndex = np.random.randint(len(choose))
                state[available[choose[0][chooseIndex],choose[1][chooseIndex]]] = self.id
        return state
    #值更新
    def valueUpdate(self,State):
        self.currentState = State.copy()
        self.value[tuple(self.previousState.astype)] += self.alpha*(self.value[tuple(self.currentState.astype(int))])
        self.value[tuple(self.previousState.astype(int))]
        self.previousState = self.currentState.copy()

class Agent():
    # 初始化
    def __init__(self,ID,epsilon=0.1,learn_rate=0.1):
        self.id = ID 
        self.epsilon = epsilon #随机率
        self.learn_rate = learn_rate #学习速率
        self.value = dict()  #状态价值字典
        self.reward = 0
        self.state_now = np.zeros(9) #当前状态
        self.state_before = np.zeros(9) #之前状态
    # 下棋
    def play(self,state):
        random_num = np.random.uniform(0,1)
        if random_num <= self.epsilon: #随机下棋
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
                if long_state[i] == 0:
                    long_state[i] = self.id
                    val = self.value.get(tuple(long_state))
                    if val > max_value:
                        max_value = val
                        max_state.clear
                        max_state.append(long_state)
                    elif val == max_value:
                        max_state.append(long_state)
                    long_state[i] = 0
            long_state = random.choice(max_state)
            return long_state.reshape((3, 3))

                

    # 更新状态价值
    def update_value(self,state,reward):
        long_state = state.flatten()
        if self.value.get(tuple(long_state)) == None:
            self.value[tuple(long_state)] = 3
        else:
            val = self.value.get(tuple(state.flatten()))
            val = val + self.learn_rate*(reward + self.next_value(long_state) - val)
            self.value[tuple(state.flatten())] = val
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
        self.state_now  = np.zeros((3,3))
        self.state_before =  np.zeros((3,3))