from tkinter import *
import numpy as np

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
