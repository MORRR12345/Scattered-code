from config import map, algorithm
import numpy as np

class TAPenv():

    # 初始化TAPenv类的实例。
    def __init__(self):
        self.num_task = map.num_task
        self.task_pos = np.load('data/task_pos.npy')
        self.agent_pos = np.zeros((2,1))
        self.c = (map.size_x**2 + map.size_y**2)**0.5

    def reset(self):
        state = np.zeros((3,self.num_task+1))
        state[1:3, :] = np.hstack((map.agent_pos, self.task_pos))
        return state
    
    def step(self, state, action):
        next_state = state.copy()
        # 判断距离
        if np.all(state[0, :] == 0): #出发第一步
            distance = np.linalg.norm(state[1:3, action] - state[1:3, 0])
        else: #不是出发第一步
            last_action = np.argwhere(state[0, :] == 2)[0][0]
            distance = np.linalg.norm(state[1:3, last_action] - state[1:3, action])
            next_state[0, last_action] = 1
        # 计算奖励
        if next_state[0, action] == 0: # 没去过
            reward = self.c - distance
        else:
            reward = - distance - 4.0# 去过
        next_state[0, action] = 2
        done = next_state[0, 0]
        return next_state, reward, done
        
def main():
    return 0

if __name__ == "__main__":
    main()