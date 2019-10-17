import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from numba import njit
import os

from net import Net


@njit
def discount_and_norm_rewards(ep_rs, gamma):
    # 折扣回报和
    discounted_ep_rs = np.zeros_like(ep_rs)
    running_add = 0
    for t in range(len(ep_rs) - 1, -1, -1):
        running_add = running_add * gamma + ep_rs[t]
        discounted_ep_rs[t] = running_add
    # 归一化
    discounted_ep_rs -= np.mean(discounted_ep_rs)
    discounted_ep_rs /= np.std(discounted_ep_rs)
    return discounted_ep_rs


class PolicyGradient(object):
    def __init__(self, n_actions, n_features,
                 learning_rate=0.02, reward_decay=0.95):
        self.n_actions = n_actions  # 动作空间的维数
        self.n_features = n_features  # 状态特征的维数
        self.lr = learning_rate  # 学习速率
        self.gamma = reward_decay  # 回报衰减率
        # 一条轨迹的观测值，动作值，和回报值
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        # 创建策略网络
        self.net = Net().cuda()
        self.optim = optim.Adam(self.net.parameters(), self.lr)

    def choose_action(self, observation: np.ndarray):
        prob_weights: np.ndarray = self.net(torch.from_numpy(
            observation[np.newaxis, :]).float().cuda())
        prob_weights = prob_weights.cpu().detach().numpy()
        # 按照给定的概率采样
        action = np.random.choice(
            range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def greedy(self, observation: np.ndarray):
        prob_weights: np.ndarray = self.net(
            torch.from_numpy(observation).float().cuda())
        prob_weights = prob_weights.cpu().detach().numpy()
        action = np.argmax(prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        """
        定义存储，将一个回合的状态，动作和回报都保存在一起
        """
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
    # 学习，以便更新策略网络参数，一个episode之后学一回

    def learn(self):
        # 计算一个episode的折扣回报
        discounted_ep_rs_norm = discount_and_norm_rewards(np.array(self.ep_rs), self.gamma)
        # 调用训练函数更新参数
        obs = np.vstack(self.ep_obs)
        acts = torch.from_numpy(np.array(self.ep_as)).long().cuda()
        vt = torch.from_numpy(discounted_ep_rs_norm).float().cuda()

        self.optim.zero_grad()
        outputs = self.net(torch.from_numpy(obs).float().cuda())
        loss = self._loss(outputs, acts, vt)
        loss.backward()
        self.optim.step()

        # 清空episode数据
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return loss.detach().cpu().item()

    def load_net(self):
        self.net.load_state_dict(torch.load("checkpoints/best.pt"))

    def save_net(self):
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(self.net.state_dict(), "checkpoints/best.pt")

    def _loss(self, outputs, acts, vt):
        neg_log_prob: torch.Tensor = F.cross_entropy(outputs, acts, reduce=False)
        loss = (neg_log_prob * vt).mean()
        return loss
