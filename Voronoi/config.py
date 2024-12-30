import numpy as np

# 定义地图参数
class map():
    # 地图参数
    size_x = 10.
    size_y = 10.
    # 任务参数
    num_task = 20
    size_task = 18.
    # 智能体参数
    num_agent = 1
    size_agent = 28.
    agent_v = 0.4
    agent_pos = np.zeros((2, num_agent))
    # 帧率
    frame_rate = 20

# 定义算法参数
class algorithm():
    linear = [12, 12]
    episodes = 1000 # 一次课程迭代数量
    random_rate = 0.4 # 随机率
    resume = False # 接着上次训练
    train_course = 3 # 课程数
    seed = 0 # 随机数种子
    gamma = 1.0
    # 奖励参数设置：
    scale_finish = 1.0
    scale_nowdis = 1.0
    scale_alldis = 0.0
    scale_nofinsh = 1.0