import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque
import gym
from gym import spaces, ActionWrapper
from timeit import default_timer as timer


# 定义自定义航天器追逃博弈环境
class SpacecraftChaseEnv(gym.Env):
    def __init__(self):
        super(SpacecraftChaseEnv, self).__init__()

        # 定义状态空间和动作空间
        self.state_dim = 6  # 追踪者和被追者的 x, y, z 坐标
        self.action_dim = 2  # 追踪者和被追者的欧拉角
        self.max_action = 1.0  # 每步的最大速度

        self.action_space = spaces.Box(low=-self.max_action, high=self.max_action, shape=(self.action_dim,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        self.reset()

    def reset(self):
        # 初始化追逃者和被追者的位置
        self.chaser_position = [-5, 10, 5]  # 追逃者位置和速度
        self.evader_position = [10, -5, -5]  # 被追者位置和速度
        self.chaser_velocity = [0, 0, 0]
        self.evader_velocity = [0, 0, 0]
        self.state = np.concatenate([self.chaser_position, self.evader_position])
        return self.state

    def step(self, action_p, action_e, sampling_time, tm):
        # 限制动作范围
        # action = np.clip(action, -self.max_action, self.max_action)

        # 更新追逃者的位置

        n0 = np.sqrt(3.986004418 * 10 ** 14 / 42249137 ** 3)

        u_p = [np.cos(action_p[0]) * np.cos(action_p[1]), np.sin(action_p[0]) * np.cos(action_p[1]), np.sin(action_p[1])]

        v0_p = self.chaser_velocity[0] + 3 * n0 ** 2 * sampling_time * self.chaser_position[0] + 2 * n0 * sampling_time * \
                 self.chaser_velocity[1] + sampling_time * u_p[0]
        v0_p = np.squeeze(v0_p)
        v1_p = self.chaser_velocity[1] - 2 * n0 * sampling_time * self.chaser_velocity[0] + sampling_time * u_p[1]
        v1_p = np.squeeze(v1_p)
        v2_p = self.chaser_velocity[2] - n0 ** 2 * sampling_time * self.chaser_position[2] + sampling_time * u_p[2]
        v2_p = np.squeeze(v2_p)

        self.chaser_velocity[0] = v0_p
        self.chaser_velocity[1] = v1_p
        self.chaser_velocity[2] = v2_p

        self.chaser_position[0] += sampling_time * self.chaser_velocity[0]
        self.chaser_position[1] += sampling_time * self.chaser_velocity[1]
        self.chaser_position[2] += sampling_time * self.chaser_velocity[2]

        self.chaser_position = np.array(self.chaser_position)

        # 模拟被追者随机移动
        # evader_move = np.random.uniform(0, 0, size=(3,))
        # self.evader_position += evader_move

        u_e = [0.3 * np.cos(action_e[0]) * np.cos(action_e[1]), 0.3 * np.sin(action_e[0]) * np.cos(action_e[1]), 0.3 * np.sin(action_e[1])]

        v0_e = self.evader_velocity[0] + 3 * n0 ** 2 * sampling_time * self.evader_position[0] + 2 * n0 * sampling_time * \
                 self.evader_velocity[1] + sampling_time * u_e[0]
        v0_e = np.squeeze(v0_e)
        v1_e = self.evader_velocity[1] - 2 * n0 * sampling_time * self.evader_velocity[0] + sampling_time * u_e[1]
        v1_e = np.squeeze(v1_e)
        v2_e = self.evader_velocity[2] - n0 ** 2 * sampling_time * self.evader_position[2] + sampling_time * u_e[2]
        v2_e = np.squeeze(v2_e)

        self.evader_velocity[0] = v0_e
        self.evader_velocity[1] = v1_e
        self.evader_velocity[2] = v2_e

        self.evader_position[0] += sampling_time * self.evader_velocity[0]
        self.evader_position[1] += sampling_time * self.evader_velocity[1]
        self.evader_position[2] += sampling_time * self.evader_velocity[2]

        self.evader_position = np.array(self.evader_position)

        self.state = np.concatenate([self.chaser_position, self.evader_position])
        distance = np.linalg.norm(self.chaser_position - self.evader_position)

        # 追逃双方奖励设计
        reward_p = -np.exp(1e-2 * tm)
        reward_e = np.exp(1e-2 * tm)

        # 限制边界
        if np.any(np.abs(self.chaser_position) > 100) or np.any(np.abs(self.evader_position) > 100):
            reward_p += -500  # 边界惩罚
            reward_e += 500

        if distance < 1.0:
            reward_p = 10000
            reward_e = -10000
            done = True
            print(reward_p, tm, distance)
        else:
            # 新增基于距离的奖励和追踪角度的奖励
            angle_reward = np.dot(self.chaser_velocity, self.evader_position - self.chaser_position) / (
                        np.linalg.norm(self.chaser_velocity) * distance) # action和P、E间位移夹角的cos值

            reward_p += -np.exp(2 * 1e-3 * distance) + angle_reward * 50  # angle_reward 系数可调整
            reward_e += np.exp(2 * 1e-3 * distance) - angle_reward * 50
            done = False

        if np.any(np.abs(self.chaser_position) > 100) or np.any(np.abs(self.evader_position) > 100):
            done = True
            print(f"P's reward: {reward_p}, Time: {tm}, Distance: {distance}")
        

        return self.state, reward_p, reward_e, done, {}
    
        

    def render(self, mode='human'):
        print(f"Chaser Position: {self.chaser_position}, Evader Position: {self.evader_position}")

    def close(self):
        pass


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_init=None):
        # self.x_prev = None
        self.theta = theta
        self.mean = mean
        self.std_deviation = std_deviation
        self.dt = dt
        self.x_init = x_init
        self.reset()

    def __call__(self):
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_deviation * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )

        self.x_prev = x
        return x

    def reset(self):
        if self.x_init is not None:
            self.x_prev = self.x_init
        else:
            self.x_prev = np.zeros_like(self.mean)


# 定义Actor网络
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(6, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * 1.0
        return action


# 定义Critic网络
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(6 + 2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


# DDPG智能体
class DDPGAgent:
    def __init__(self):
        self.actor = Actor()
        self.actor_target = Actor()
        self.critic = Critic()
        self.critic_target = Critic()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.memory = deque(maxlen=200000)
        self.update_target_networks()

    def update_target_networks(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

    def select_action(self, state, noise):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        action = action + noise
        return action

    def store_transition(self, transition):
        self.memory.append(transition)

    def train(self):
        if len(self.memory) < 64:
            return
        batch = random.sample(self.memory, 64)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(next_states, next_actions)
        q_targets = rewards + 0.99 * (1 - dones) * next_q_values
        q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_values, q_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_networks()


# 主训练循环
env = SpacecraftChaseEnv()  # 使用自定义环境
agent_p = DDPGAgent()
agent_e = DDPGAgent()
num_episodes = 1000
rewardp = 0
rewarde = 0
sampling_time = 0.1

for episode in range(num_episodes):

    std_dev_p = 5 * (1 - np.exp(-(5000 - rewardp)))
    std_dev_e = 2
    # init trajectory vectors
    xp_pos = []
    yp_pos = []
    zp_pos = []
    xe_pos = []
    ye_pos = []
    ze_pos = []

    state = env.reset()

    xp_pos.append(state[0])
    yp_pos.append(state[1])
    zp_pos.append(state[2])
    xe_pos.append(state[3])
    ye_pos.append(state[4])
    ze_pos.append(state[5])

    times = 0
    episode_reward = 0
    done = False
    while not done:
        times += sampling_time
        p_ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev_p) * np.ones(1), theta=0.15, dt=1e-2,
                                   x_init=None)
        e_ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev_e) * np.ones(1), theta=0.15, dt=1e-2,
                                   x_init=None)
        
        noise_ob1 = p_ou_noise()
        noise_ob2 = p_ou_noise()
        noise_obp = np.concatenate([noise_ob1, noise_ob2])
        actionp = agent_p.select_action(state, noise_obp)

        noise_ob3 = e_ou_noise()
        noise_ob4 = e_ou_noise()
        noise_obe = np.concatenate([noise_ob3, noise_ob4])
        actione = agent_e.select_action(state, noise_obe)

        next_state, rewardp, rewarde, done, _ = env.step(actionp, actione, sampling_time, times)
        agent_p.store_transition((state, actionp, rewardp, next_state, done))
        agent_p.train()
        agent_e.store_transition((state, actione, rewarde, next_state, not done))
        state = next_state
        episode_reward += rewardp

        xp_pos.append(state[0])
        yp_pos.append(state[1])
        zp_pos.append(state[2])
        xe_pos.append(state[3])
        ye_pos.append(state[4])
        ze_pos.append(state[5])

    print(f"Episode {episode}: Total Reward: {episode_reward}")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xp_pos, yp_pos, zp_pos, label='p')
    ax.plot(xe_pos, ye_pos, ze_pos, label='e')
    ax.plot(xp_pos[0], yp_pos[0], zp_pos[0], 'o')
    ax.plot(xe_pos[0], ye_pos[0], ze_pos[0], 'o')
    ax.legend()
    plt.show(block=False)
    if rewardp==5000:
        plt.pause(1000)
    plt.pause(1)
    plt.close()
    
    # user_input = input("按回车键继续...")

env.close()
