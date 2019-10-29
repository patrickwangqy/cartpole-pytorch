import gym
import timeit

from tb_log import TBLog
from pg import PolicyGradient


def train(epochs=1000000, N=32, render=False, max_step=2000, early_stop=20):
    logger = TBLog("cartpole")

    env = gym.make("CartPole-v0")
    env = env.unwrapped

    rl = PolicyGradient(
        learning_rate=0.02,
        reward_decay=0.99
    )

    best_reward = 0

    # 学习过程
    rl.net.train()
    running_reward = None
    begin_time = timeit.default_timer()
    for epoch in range(1, epochs+1):
        for i_episode in range(1, N+1):
            observation = env.reset()
            rl.new_episode()
            step = 1
            while True:
                if render:
                    env.render()
                # 采样动作，探索环境
                action = rl.choose_action(observation)

                observation_, reward, done, info = env.step(action)

                # 将观测，动作和回报存储起来
                rl.store_transition(observation, action, reward)
                if done or step >= max_step:
                    break

                # 智能体探索一步
                observation = observation_
                step += 1
        # 更新网络
        loss, reward = rl.learn()
        end_time = timeit.default_timer()
        if reward >= best_reward:
            rl.save_net()
            best_reward = reward
        if running_reward is None:
            running_reward = reward
        else:
            running_reward = running_reward * 0.99 + reward * 0.01

        print(f"epoch:{epoch}, used_time:{end_time-begin_time:.3f}s, loss:{loss:.6f}, reward:{running_reward:.6f}, current reward:{reward:.6f}")
        logger.line("rewards", reward)
        logger.line("running rewards", running_reward)
        logger.line("loss", loss)

        begin_time = timeit.default_timer()
    env.close()


if __name__ == '__main__':
    train(render=False)
