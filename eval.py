# %%
import gym

from rl import PolicyGradient


env = gym.make("CartPole-v0")
env = env.unwrapped

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99
)

RL.load_net()
RL.net.eval()
observation = env.reset()
count = 0
while True:
    # 采样动作，探索环境
    env.render()
    action = RL.greedy(observation)
    print(action)
    observation_, reward, done, info = env.step(action)
    if done:
        print(count)
        break
    observation = observation_
    count += 1
    print(count)


# %%
env.close()


# %%
