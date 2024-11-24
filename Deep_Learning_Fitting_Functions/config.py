import math
import numpy as np

class cfg():
    linear = [32, 12] # 神经网络层
    episodes = 2000 # 训练次数
    training_area = [-2, 2] # 训练范围

    # 假设的函数
    def function(x):
        y = math.exp(x)
        return y 
    output_area = np.arange(-2, 2, 0.1) # 输出范围