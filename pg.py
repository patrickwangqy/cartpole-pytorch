import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from typing import List, Any

from policy_net import Net


class PolicyGradient(object):
    def __init__(self, N=32, learning_rate=0.02, reward_decay=0.95):
        self.N = N  # 每个策略采样多少次
        self.lr = learning_rate  # 学习速率
        self.gamma = reward_decay  # 回报衰减率
        # 一条轨迹的观测值，动作值，和回报值
        self.episode_num = 0
        self.episodes_observations: List[List[Any]] = []
        self.episodes_actions: List[List[float]] = []
        self.episodes_rewards: List[List[float]] = []
        # 创建策略网络
        self.net = Net().cuda()
        self.optim = optim.Adam(self.net.parameters(), self.lr)

    def choose_action(self, observation: np.ndarray):
        with torch.no_grad():
            prob_weights: np.ndarray = self.net(torch.from_numpy(observation[np.newaxis, :]).float().cuda())
        prob_weights = prob_weights.cpu().detach().numpy()
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def greedy(self, observation: np.ndarray):
        with torch.no_grad():
            prob_weights: np.ndarray = self.net(torch.from_numpy(observation).float().cuda())
        prob_weights = prob_weights.cpu().detach().numpy()
        action = np.argmax(prob_weights.ravel())
        return action

    def new_episode(self):
        self.episodes_observations.append([])
        self.episodes_actions.append([])
        self.episodes_rewards.append([])
        self.episode_num += 1

    def store_transition(self, s, a, r):
        """
        定义存储，将一个回合的状态，动作和回报都保存在一起
        """
        self.episodes_observations[-1].append(s)
        self.episodes_actions[-1].append(a)
        self.episodes_rewards[-1].append(r)

    def learn(self):
        self.optim.zero_grad()
        loss = 0
        for i in range(self.N):
            discounted_rewards = self._calc_episode_discount_rewards(i, is_norm=True)
            obs = torch.from_numpy(np.array(self.episodes_observations[i])).float().cuda()
            acts = torch.from_numpy(np.array(self.episodes_actions[i])).long().cuda()
            vt = torch.from_numpy(discounted_rewards).float().cuda()
            outputs = self.net(obs)
            loss += self._loss(outputs, acts, vt)
        loss /= self.N
        loss.backward()
        self.optim.step()

        reward = self.avg_rewards()

        self.episodes_observations: List[List[Any]] = []
        self.episodes_actions: List[List[float]] = []
        self.episodes_rewards: List[List[float]] = []
        self.episode_num = 0

        return loss.detach().cpu().item(), reward

    def load_net(self):
        self.net.load_state_dict(torch.load("checkpoints/best.pt"))

    def save_net(self):
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(self.net.state_dict(), "checkpoints/best.pt")

    def avg_rewards(self):
        r = 0
        for i in range(self.N):
            r += sum(self.episodes_rewards[i])
        return r / self.N

    def _loss(self, outputs, acts, vt):
        mask = torch.zeros_like(outputs, dtype=torch.bool)
        mask.scatter_(-1, acts.unsqueeze(dim=-1), True)
        loss = -torch.log(outputs[mask]) * vt
        loss = loss.sum()
        # neg_log_prob: torch.Tensor = F.cross_entropy(outputs, acts, reduce=False)
        # loss = (neg_log_prob * vt).mean()
        return loss

    def _calc_episode_discount_rewards(self, index, is_norm=True):
        episode_rewards = np.array(self.episodes_rewards[index])
        # 折扣回报和
        episode_discounted_rewards = np.zeros_like(episode_rewards)
        running_add = 0
        for t in range(len(episode_rewards) - 1, -1, -1):
            running_add = running_add * self.gamma + episode_rewards[t]
            episode_discounted_rewards[t] = running_add
        if is_norm:
            # 归一化
            episode_discounted_rewards -= np.mean(episode_discounted_rewards)
            episode_discounted_rewards /= np.std(episode_discounted_rewards)
        return episode_discounted_rewards
