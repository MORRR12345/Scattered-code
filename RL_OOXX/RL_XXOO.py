from tkinter import *
import numpy as np

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
class Agent():
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
            random = np.random.uniform(0,1)
            if random < self.epsilon: #随机策略
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
        self.value[tuple(self.previousState.astype)] += \
        self.alpha*(self.value[tuple(self.currentState.astype(int))])
        self.value[tuple(self.previousState.astype(int))]
        self.previousState = self.currentState.copy()

#-----------------------机器人-----------------------
class Robot():
    #初始化
    def __init__(self,ID):
        self.id = ID
    #行动
    def play(self,State):
        state = State.copy()
        available = np.where(state==0)
        length = len(available[0])
        #print(length)
        if length > 0: #如果没有下完
            choose = np.random.randint(length)
            state[available[0][choose],available[1][choose]] = self.id
        return state

#-----------------------棋盘-----------------------
class Board():
    # 初始化
    def __init__(self):
        self.state = np.zeros((3,3))
        self.score = np.zeros(3) 
        self.size_of_board = 300 #棋盘大小
        self.symbol_size = (self.size_of_board/3 - self.size_of_board/8) / 2 #棋子大小
        self.symbol_thickness = self.size_of_board/20 #棋子粗细
        self.X_color = '#EE4035'
        self.O_color = '#0492CF'
    def reset(self):
        self.state = np.zeros((3,3))
    # 游戏是否结束
    def is_gameover(self):
        for i in range(3):
            if self.state[i][0] == self.state[i][1] == self.state[i][2] != 0:
                return self.state[i][0]
            if self.state[0][i] == self.state[1][i] == self.state[2][i] != 0:
                return self.state[0][i]
        if self.state[0][0] == self.state[1][1] == self.state[2][2] != 0:
            return self.state[0][0]
        if self.state[0][2] == self.state[1][1] == self.state[2][0] != 0:
            return self.state[0][2]
        if np.any(self.state == 0):
            return -1 #还没下完
        else:
            return 0 #平局
    def show_board(self):
        print(self.state)
    # 展示结果
    def show_outcome(self):
        print("平局：", self.score[0])
        print("玩家1：", self.score[1])
        print("玩家2：", self.score[2])

#-----------------------主函数-----------------------
def game(match_number):
    board = Board()
    player1 = Robot(1)
    player2 = Agent(2)
    # 训练迭代
    for n in range(match_number):
        finish = -1
        while finish < 0:
            board.state = player1.play(board.state)
            board.show_board()
            board.state = player2.play(board.state)
            board.show_board()
            finish = board.is_gameover()
        board.score[int(finish)] += 1
        board.reset()
    # 展示结果
    board.show_outcome()
    
if __name__ == '__main__':
    game(1)