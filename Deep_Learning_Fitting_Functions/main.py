import random
import numpy as np
from tqdm import tqdm
from config import cfg
from net import Deep
import matplotlib.pyplot as plt

def train():
    Deep_L = Deep(1, 1) # 一元函数
    loss_log = np.zeros(cfg.episodes)
    for episode in tqdm(range(cfg.episodes)):
        # 随机选择x
        x = random.uniform(cfg.training_area[0], cfg.training_area[1])
        # 计算函数值y
        y = cfg.function(x)
        # 学习并记录loss
        loss_log[episode] = Deep_L.learn(x, y)
    
    plt.plot(range(cfg.episodes), loss_log, label='loss')
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.show()
    plt.savefig('training')

    y_get = cfg.output_area.copy()
    y_target = cfg.output_area.copy()
    for i in range(len(cfg.output_area)):
        x = cfg.output_area[i]
        y_get[i] = Deep_L.get_y(x)
        y_target[i] = cfg.function(x)
    plt.plot(cfg.output_area, y_get, label='y_get')
    plt.plot(cfg.output_area, y_target, label='y_target')
    plt.xlabel('time')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    plt.savefig('output')
    
if __name__ == "__main__":
    train()