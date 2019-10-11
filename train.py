import gym
from visdom_log import VisdomLog

from rl import PolicyGradient


def train(episodes=2000, render=False, max_step=2000, early_stop=5):
    logger = VisdomLog("cartpole")

    env = gym.make("CartPole-v0")
    env = env.unwrapped

    RL = PolicyGradient(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        learning_rate=0.02,
        reward_decay=0.99
    )

    best_reward = 0

    # 学习过程
    RL.net.train()
    running_reward = None
    finish_count = 0
    for i_episode in range(episodes):
        observation = env.reset()
        step = 1
        while True:
            if render:
                env.render()
            # 采样动作，探索环境
            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            # 将观测，动作和回报存储起来
            RL.store_transition(observation, action, reward)
            if done or step >= max_step:
                ep_rs_sum = sum(RL.ep_rs)
                logger.line("rewards", ep_rs_sum)
                if ep_rs_sum >= best_reward:
                    RL.save_net()
                    best_reward = ep_rs_sum
                if running_reward is None:
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                print(f"episode:{i_episode}, rewards:{int(running_reward)}, current rewards:{int(ep_rs_sum)}")
                # 每个episode学习一次
                _vt = RL.learn()
                # if i_episode == 0:
                #     plt.plot(vt)
                #     plt.xlabel('episode steps')
                #     plt.ylabel('normalized state-action value')
                #     plt.show()
                break

            # 智能体探索一步
            observation = observation_
            step += 1
        if step >= max_step:
            finish_count += 1
            if finish_count >= early_stop:
                break
        else:
            finish_count = 0
    env.close()


if __name__ == '__main__':
    train(render=False)
