import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import Animation
from config import map, algorithm
from dqn import DQNAgent
from TAP_env import TAPenv

# AI行动
def AI_action(state, random_rate):
    agent = DQNAgent((map.num_task+1)*3, map.num_task+1, 1000, random_rate) # AI机器人
    agent.read_model()
    action = agent.choose_action(state)
    return action

def play():

    # 初始化agent对象
    num_agent = map.num_agent
    agent_pos = map.agent_pos
    agent_color = np.zeros((3, num_agent))
    agent_v = map.agent_v

    # 初始化任务对象
    num_task = map.num_task
    task_pos = np.load('data/task_pos.npy')
    task_color = np.zeros((3, num_task))

    # 初始化图像
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-map.size_x/2, map.size_x/2)
    ax.set_ylim(-map.size_y/2, map.size_y/2)
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)

    # 创建散点图对象，用于在动画中更新
    scat_agent = ax.scatter(agent_pos[0], agent_pos[1], s=map.size_agent, c='r')
    scat_task = ax.scatter(task_pos[0], task_pos[1], s=map.size_task, c='b')
    ax.set_title('Deep_RL')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    env = TAPenv()
    state = env.reset()
    done = 0
    num_finished = 0
    time = 0
    # 初始状态
    while done == 0:
        # 选择行动
        action = AI_action(state, 0.0)
        next_state,_,done = env.step(state, action)
        # 初始化
        arrive = False
        end_pos = state[1:3, action].reshape(2, 1)
        # 更新移动动画
        while not arrive:
            distance = np.linalg.norm(agent_pos - end_pos)
            if distance < agent_v:
                arrive = True
                num_finished += 1
                if action >= 1:
                    task_color[:, action-1] = np.array([0, 0, 1]).T
                agent_pos = end_pos.copy()
            else:
                agent_pos += agent_v * (end_pos - agent_pos) / distance
            plt.plot(agent_pos[0], agent_pos[1], 'ro', markersize=2, color='gray', alpha=0.2)  # 绘制agent
            scat_agent.set_offsets(agent_pos.T)
            scat_task.set_color(task_color.T)
            plt.pause(0.02)  # 暂停0.02秒,50帧
            time += 1
        state = next_state

    print(f'Total time: {time} seconds')
    print(f'Percentage of completed tasks: {(num_finished/num_task)*100:.2f}%')
    # 显示动画
    plt.show()
    plt.close()

if __name__ == "__main__":
    play()