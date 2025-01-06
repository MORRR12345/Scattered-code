import gym
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    print('开始训练！')
    env = gym.make('CartPole-v0')
    # 每经过N步就更新一次网络
    N = 20
    batch_size = 5
    # 每次更新的次数
    n_epochs = 4
    # 学习率
    alpha = 0.0003
    # 初始化智能体
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                    alpha=alpha, n_epochs=n_epochs,
                    input_dims=env.observation_space.shape)
    # 训练轮数
    n_games = 300

    # 统计图
    figure_file = 'plots/cartpole.png'
    # 存储最佳得分
    best_score = env.reward_range[0]
    # 存储历史分数
    score_history = []
    # 更新网络的次数
    learn_iters = 0
    # 每一轮的得分
    avg_score = 0
    # 总共在环境中走的步数
    n_steps = 0

    # 开始玩游戏
    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            env.render()
            n_steps += 1
            score += reward
            # 存储轨迹
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                # 更新网络
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        # 比较最佳得分  保存最优的策略
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)