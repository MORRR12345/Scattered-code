from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
#-----------------------棋盘-----------------------
class Board():

    # 初始化
    def __init__(self):
        self.state = np.zeros((3,3))
        self.score = np.zeros(3) 
        self.score_records_time = []
        self.score_records_draw = []
        self.score_records_1win = []
        self.score_records_2win = []
        self.size_of_board = 300 #棋盘大小
        self.symbol_size = (self.size_of_board/3 - self.size_of_board/8) / 2 #棋子大小
        self.symbol_thickness = self.size_of_board/20 #棋子粗细
        self.X_color = '#EE4035'
        self.O_color = '#0492CF'

    # 重置棋盘
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
        
    # 展示棋盘
    def show_board(self):
        print(self.state)

    # 计分
    def scoring(self,win_ID):
        if win_ID >= 0: 
            self.score[int(win_ID)] += 1
            self.score_records_time.append(len(self.score_records_time)+1)
            self.score_records_draw.append(self.score[0]/len(self.score_records_time))
            self.score_records_1win.append(self.score[1]/len(self.score_records_time))
            self.score_records_2win.append(self.score[2]/len(self.score_records_time))
        
    # 展示结果
    def show_outcome(self):
        print("平局：", self.score[0])
        print("玩家1：", self.score[1])
        print("玩家2：", self.score[2])
        plt.plot(self.score_records_time, self.score_records_draw, label='draw')
        plt.plot(self.score_records_time, self.score_records_1win, label='player1')
        plt.plot(self.score_records_time, self.score_records_2win, label='player2')
        plt.legend()
        plt.show()
