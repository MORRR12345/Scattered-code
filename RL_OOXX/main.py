from tkinter import *
import numpy as np
from tqdm import tqdm
import environment
import robot
import agent

#-----------------------主函数-----------------------
def game(match_number):
    board = environment.Board()
    player1 = agent.Agent(1,1,0) # ID，随机概率，学习率
    player2 = agent.Agent(2,0,0.3) # ID，随机概率，学习率
    # 训练迭代
    for n in tqdm(range(match_number)):
        #if n > 0.67*match_number:
        #    player1.epsilon = 0
        #    player2.epsilon = 0
        win_ID = -1 #开始一局比赛
        first_player = 1
        while win_ID < 0:
            if first_player == 1:
                board.state = player1.play(board.state)
                first_player = 2
            else:
                board.state = player2.play(board.state)
                first_player = 1
            win_ID = board.is_gameover()
            player1.update_value(board.state,win_ID) #学习
            player2.update_value(board.state,win_ID) #学习
        board.scoring(win_ID)
        board.reset()
        player1.reset()
        player2.reset()
    # 展示结果
    board.show_outcome()

if __name__ == '__main__':
    game(10000)