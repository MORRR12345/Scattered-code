import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import Animation
from config import map, algorithm

def K_means():
    # 初始化中心圆点
    num_center = map.num_agent
    center_pos = np.zeros((2, num_center))
    center_color = np.zeros((3, num_center))
    center_color[0] = 1 #红色
    # 为每个聚类赋予一个随机颜色
    center_colors = np.random.uniform(0, 1, (3, num_center))

    # 初始化任务对象
    num_task = map.num_task
    task_pos = np.load('data/task_pos.npy')
    task_color = np.zeros((3, num_task))
    
    # 归属矩阵
    belong = np.zeros((num_center, num_task))
    belong = init_belong(belong, num_center, num_task)
    belong_p = np.zeros((num_center, num_task))

    # 初始化图像
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-map.size_x/2, map.size_x/2)
    ax.set_ylim(-map.size_y/2, map.size_y/2)
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)
    # 创建散点图对象，用于在动画中更新
    scat_center = ax.scatter(center_pos[0], center_pos[1], s=map.size_agent, c='r')
    scat_task = ax.scatter(task_pos[0], task_pos[1], s=map.size_task, c='b')
    ax.set_title('K-Means')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # K-Means 聚类
    while not np.array_equal(belong_p, belong):
        belong_p = belong.copy()
        for i in range(num_center):
            center_pos[:, i] = np.average(task_pos, axis=1, weights=belong[i, :])
        distances = dis(center_pos, task_pos)
        for i in range(num_task):
            min_pos = np.argmin(distances[:, i])
            belong[:, i] = 0
            belong[min_pos, i] = 1
            task_color[:, i] = center_colors[:, min_pos]
        scat_center.set_offsets(center_pos.T)
        scat_task.set_color(task_color.T)
        plt.pause(0.5)  # 暂停0.5秒

    # 显示动画
    plt.show()

def dis(a, b):
    distances = np.zeros((a.shape[1], b.shape[1]))
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            distances[i, j] = np.linalg.norm(a[:, i] - b[:, j])
    return distances

def init_belong(belong, x, y):
    for i in range(x):
        for j in range(y):
            if i*y/x <= j and j < (i+1)*y/x:
                belong[i, j] = 1
    return belong

if __name__ == "__main__":
    K_means()