import matplotlib.pyplot as plt
import numpy as np
from config import map, algorithm

# 随机生成任务
def create_task():
    num_task = map.num_task
    task_pos = np.random.uniform(-1, 1, (2, num_task))
    task_pos[0] = task_pos[0]*map.size_x/2
    task_pos[1] = task_pos[1]*map.size_y/2
    np.save('data/task_pos.npy', task_pos)

# 在地图上绘制任务
def show_task():
    # 创建一个新的图形
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)
    task_pos = np.load('data/task_pos.npy')
    # print(task_pos)
    # 绘制圆
    scat_task = ax.scatter(task_pos[0], task_pos[1], s=map.size_task, c='b')

    # 设置坐标轴的范围
    ax.set_xlim(-map.size_x/2, map.size_x/2)
    ax.set_ylim(-map.size_y/2, map.size_y/2)


    # 显示图形
    plt.show()

if __name__ == "__main__":
    create_task()
    show_task()