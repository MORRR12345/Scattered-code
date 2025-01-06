import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:
    """
    经验池
    """
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    """
    构建策略网络--actor
    """
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='tmp'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
        返回动作的概率分布
        :param state:
        :return:
        """
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist  # 返回动作的概率分布

    def save_checkpoint(self):
        """
        保存模型
        :return:
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        加载模型
        :return:
        """
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    """
    构建价值网络--critic
    """
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir='tmp'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        """
        保存模型
        :return:
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        加载模型
        :return:
        """
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        # 实例化策略网络
        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        # 实例化价值网络
        self.critic = CriticNetwork(input_dims, alpha)
        # 实例化经验池
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        """
        记录轨迹
        :param state:
        :param action:
        :param probs:
        :param vals:
        :param reward:
        :param done:
        :return:
        """
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        """
        选择动作
        :param observation:
        :return:
        """
        # 维度变换 [n_state]-->tensor[1,n_states]
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        # 当前状态下，每个动作的概率分布 [1,n_states]
        dist = self.actor(state)
        # 预测，当前状态的state_value  [b,1]
        value = self.critic(state)
        # 依据其概率随机挑选一个动作
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        # 每次学习需要更新n_epochs次参数
        for _ in range(self.n_epochs):
            # 提取数据集
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            # 计算优势函数
            for t in range(len(reward_arr) - 1): # 逆序时序差分值 axis=1轴上倒着取 [], [], []
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            # 估计状态的值函数的数组
            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                # 获取数据
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                # 用当前网络进行预测
                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)
                # 每一轮更新一次策略网络预测的状态
                new_probs = dist.log_prob(actions)
                # 新旧策略之间的比例
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                # 近端策略优化裁剪目标函数公式的左侧项
                weighted_probs = advantage[batch] * prob_ratio
                # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                # 计算损失值进行梯度下降
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()