import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import torch.optim as optim
import copy
import time
from torch.utils.tensorboard import SummaryWriter
import sys
import os
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm
from enum import Enum
curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
sys.path.append(curr_path)  # add current terminal path to sys.path
from util import color
from actions.Action import Action
from host import Host_state, HOST
from config import SAC_Config
from common import Normalization
from datetime import datetime

"""
SAC version from https://github.com/vwxyzjn/cleanrl
"""


def clamp(n, min_, max_):
    return max(min_, min(n, max_))


# pdist = nn.PairwiseDistance(p=2)
class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)


def CosineDistance(x, y):
    similarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return float(1 - similarity)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0, metric=""):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        label = torch.nn.functional.relu(label, inplace=True)

        # distance = SiameseDistanceMetric.COSINE_DISTANCE(output1, output2)
        distance = SiameseDistanceMetric.EUCLIDEAN(output1, output2)
        loss_contrastive = torch.mean(
            (label) * torch.pow(distance, 2)  # calmp夹断用法
            + (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        )

        return loss_contrastive


class ReplayBuffer(object):

    def __init__(self, state_dim, action_dim, memory_size):
        self.max_size = int(memory_size)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.p_a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, p_a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.p_a[self.count] = p_a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw

        self.count = (
            self.count + 1
        ) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(
            self.size + 1, self.max_size
        )  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_p_a = torch.tensor(self.p_a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)

        return batch_s, batch_a, batch_p_a, batch_r, batch_s_, batch_dw


class Actor(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_width,
        max_abs_action,
        max_action,
        min_action,
        activate_func="tanh",
    ):
        super(Actor, self).__init__()
        self.max_abs_action = max_abs_action
        self.max_action = max_action
        self.min_action = min_action
        self.action_scale = torch.tensor(
            (max_action - min_action) / 2, dtype=torch.float32
        )
        self.action_bias = torch.tensor(
            (max_action + min_action) / 2, dtype=torch.float32
        )

        self.l1 = nn.Linear(state_dim, 1024)
        self.norm1 = nn.LayerNorm([1024])
        self.l2 = nn.Linear(1024, hidden_width)
        self.norm2 = nn.LayerNorm([hidden_width])
        # self.l3 = nn.Linear(hidden_width, hidden_width)
        # self.norm3 = nn.LayerNorm([hidden_width])
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std_layer = nn.Linear(hidden_width, action_dim)

        if activate_func == "relu":
            self.activate_func = nn.ReLU()
        elif activate_func == "leaky_relu":
            self.activate_func = nn.LeakyReLU()
        elif activate_func == "tanh":
            self.activate_func = nn.Tanh()
        elif activate_func == "softsign":
            self.activate_func = nn.Softsign()
        elif activate_func == "tanhshrink":
            self.activate_func = nn.Tanhshrink()
        elif activate_func == "elu":
            self.activate_func = nn.ELU()
        else:
            logging.error("activate_func error")
            self.activate_func = nn.ReLU()

    def forward(self, x):
        x = self.activate_func(self.norm1(self.l1(x)))
        x = self.activate_func(self.norm2(self.l2(x)))
        # x = self.activate_func(self.l3(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(
            x
        )  # We output the log_std to ensure that std=exp(log_std)>0

        log_std = torch.clamp(log_std, -5, 2)
        # log_std = torch.tanh(log_std)
        # # From SpinUp / Denis Yarats
        # log_std = LOG_STD_MIN + 0.5 * \
        #     (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-5)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        action = F.normalize(action, p=2, dim=1)
        mean = F.normalize(mean, p=2, dim=1)
        return action, log_prob, mean


class SingleCritic(nn.Module):  # According to (s,a), directly calculate Q(s,a)

    def __init__(self, state_dim, action_dim, hidden_width, activate_func="tanh"):
        super(SingleCritic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 2048)
        self.norm1 = nn.LayerNorm([2048])
        self.l2 = nn.Linear(2048, hidden_width)
        self.norm2 = nn.LayerNorm([hidden_width])
        self.l3 = nn.Linear(hidden_width, hidden_width)
        self.norm3 = nn.LayerNorm([hidden_width])
        self.l4 = nn.Linear(hidden_width, 1)

        # activate_func == "tanh"
        if activate_func == "relu":
            self.activate_func = nn.ReLU()
        elif activate_func == "leaky_relu":
            self.activate_func = nn.LeakyReLU()
        elif activate_func == "tanh":
            self.activate_func = nn.Tanh()
        elif activate_func == "softsign":
            self.activate_func = nn.Softsign()
        elif activate_func == "tanhshrink":
            self.activate_func = nn.Tanhshrink()
        elif activate_func == "elu":
            self.activate_func = nn.ELU()
        else:
            logging.error("activate_func error")
            self.activate_func = nn.ReLU()

    def forward(self, s, a):
        s_a = torch.cat([s, a], -1)
        q1 = self.activate_func(self.norm1(self.l1(s_a)))
        q1 = self.activate_func(self.norm2(self.l2(q1)))
        q1 = self.activate_func(self.norm3(self.l3(q1)))
        # q1 = self.activate_func(self.l3(q1))
        q1 = self.l4(q1)

        return q1


class SAC_agent:
    # action_embedding = Action_embedding(actions=Action.legal_actions_name, action_path=Action.vul_hub_path)
    def __init__(self, cfg: SAC_Config):
        self.current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.name = "APRIL-AE"
        self.config = cfg
        self.batch_size = self.config.batch_size
        self.random_steps = self.config.random_step
        self.train_episodes = self.config.train_eps
        self.explore_episode = self.config.explore_eps
        self.gamma = self.config.gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.TAU = self.config.tau  # Softly update the target network
        self.logger = SummaryWriter()
        self.state_dim = Host_state.state_space
        self.num_actions = len(Action.legal_actions_name)
        self.k_nearest_neighbors = self.config.k_nearest_neighbors
        self.action_embedding = Action.action_embedding
        self.action_dim = self.action_embedding.action_dim
        self.max_abs_action = self.action_embedding.max_abs_action
        self.max_action = self.action_embedding.max_action
        self.min_action = self.action_embedding.min_action
        self.activate_func = self.config.activate_func
        self.actor = Actor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_width=self.config.hidden_sizes,
            max_abs_action=self.max_abs_action,
            max_action=self.max_action,
            min_action=self.min_action,
            activate_func=self.activate_func,
        ).to(self.device)

        self.critic_1 = SingleCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_width=self.config.hidden_sizes,
            activate_func=self.activate_func,
        ).to(self.device)
        self.critic_2 = SingleCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_width=self.config.hidden_sizes,
            activate_func=self.activate_func,
        ).to(self.device)
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        self.critic_target_2 = copy.deepcopy(self.critic_2)
        self.use_distance_loss = self.config.use_distance_loss
        if self.use_distance_loss == "ContrastiveLoss":
            # self.distance_loss = ContrastiveLoss(size_average=True,margin=2)
            self.distance_loss = ContrastiveLoss(margin=1.5)
        else:
            self.use_distance_loss = None
            self.distance_loss = None
        self.distance_loss_beta = self.config.distance_loss_beta
        # Whether to automatically learn the temperature alpha
        self.adaptive_alpha = self.config.adaptive_alpha
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = self.config.target_entropy
            # self.config.target_entropy=self.target_entropy
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp().to(self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr_alpha)
        else:
            self.alpha = torch.tensor([0.2]).to(self.device)
        self.use_grad_clip = self.config.use_grad_clip

        self.a_lr = self.config.actor_lr
        self.c_lr = self.config.critic_lr
        self.alpha_lr = self.config.lr_alpha
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=self.config.critic_lr,
        )
        self.memory_size = self.config.memory_size
        self.memory = ReplayBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            memory_size=self.memory_size,
        )

        self.loss = 0
        self.num_episodes = 0
        self.training_step = 0
        self.total_steps = 0
        self.step_limit = self.config.step_limit
        self.eval_step_limit = self.config.eval_step_limit
        self.action_set = []
        self.total_action_set = set()
        self.reward_set = []
        self.best_return = -float("inf")
        self.best_action_set = []
        self.best_episode = 0
        self.best_reward_episode = []
        self.eval_rewards = 0
        self.eval_success_rate = 0.0
        self.use_state_norm = self.config.use_state_norm
        if self.use_state_norm:
            self.state_norm = Normalization(shape=self.state_dim)
        self.explore_eps = self.config.explore_eps
        self.epsilon_schedule = np.linspace(1.0, 0.0, self.explore_eps)
        self.ucb_lamba = self.config.ucb_lamba
        self.is_loaded_agent = False
        self.policy_frequency = 1
        self.target_network_frequency = 1
        self.action_refinement_method = self.config.action_refinement
        assert self.action_refinement_method in [
            "Random",
            "UCB",
            "Greedy",
        ], "action_refinement methdo must be Random/UCB/Greedy"

    def select_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        a, _, _ = self.actor.get_action(
            s
        )  # When choosing actions, we do not need to compute log_pi
        probe_action = a.cpu().detach().numpy().flatten()

        return probe_action

    def get_epsilon(self):
        if self.num_episodes < self.explore_eps and not self.is_loaded_agent:
            return self.epsilon_schedule[self.num_episodes]
        return 0.0

    def action_refinement(self, proto_action, state, k_nearest_neighbors=None):

        # nor_probe_action=probe_action /  np.linalg.norm(probe_action, axis=0, keepdims=True)
        if not k_nearest_neighbors:
            k_nearest_neighbors = self.k_nearest_neighbors
        p_a = proto_action  # /np.linalg.norm(probe_action, ord=2)
        raw_actions, actions, action_index, distance = (
            self.action_embedding.get_nearest_neighbor(point=p_a, k=k_nearest_neighbors)
        )
        if isinstance(distance, float):
            distance = [distance]

        s_t = state
        if not isinstance(s_t, np.ndarray):
            s_t = state.cpu().data.numpy()
        if k_nearest_neighbors > 1:
            eps = self.get_epsilon()
            if random.random() <= eps or self.action_refinement_method == "Random":
                index = random.randint(0, k_nearest_neighbors - 1)
                max_index = np.array([index])
            else:
                s_t = np.tile(s_t, [raw_actions.shape[1], 1])

                s_t = s_t.reshape(len(raw_actions), raw_actions.shape[1], s_t.shape[1])

                # raw_actions = torch.from_numpy(raw_actions).to(self.device)
                s_t = torch.from_numpy(s_t).to(self.device)
                raw_actions = torch.from_numpy(raw_actions).to(self.device)
                target_Q1 = self.critic_1(s_t, raw_actions)
                target_Q2 = self.critic_2(s_t, raw_actions)
                Q1 = target_Q1.detach().cpu().data.numpy()
                Q2 = target_Q2.detach().cpu().data.numpy()
                mean = 0.5 * (Q1 + Q2)
                # mean_2=(Q1+Q2+Q3+Q4)/4
                # minQ=np.min(Q1,Q2)
                if self.action_refinement_method == "Greedy":
                    score = mean
                else:  # self.action_refinement=="UCB":
                    var = np.sqrt(0.5 * ((Q1 - mean) ** 2 + (Q2 - mean) ** 2))
                    # var_2 = np.sqrt(0.25 * ((Q1 - mean_2)**2 + (Q2 - mean_2)**2+(Q3 - mean_2)**2+(Q4 - mean_2)**2))
                    score = mean + self.ucb_lamba * var
                    # score_2 = mean_2 + self.ucb_lamba * var_2
                # evaluate each pair through the critic
                # target_Q1, target_Q2 = self.critic(s_t, raw_actions)
                # actions_evaluation = 0.5 * (target_Q1 +
                #                             target_Q2).cpu().data.numpy()

                # find the index of the pair with the maximum value
                max_index = np.argmax(score, axis=1)
                max_index = max_index.reshape(
                    len(max_index),
                )

                raw_actions = raw_actions.cpu().data.numpy()
            # return the best action, i.e., wolpertinger action from the full wolpertinger policy

            raw_wolp_action = raw_actions[
                [i for i in range(len(raw_actions))], max_index
            ]
            wolp_action = actions[[i for i in range(len(actions))], max_index]
            # raw_wolp_action=raw_actions[[i for i in range(len(raw_actions))], max_index, [0]].reshape(len(raw_actions),1)
            # wolp_action=actions[[i for i in range(len(actions))], max_index, [0]].reshape(len(actions),1)
            assert len(max_index) == 1
            action_index = action_index[max_index[0]]

        else:
            raw_wolp_action = raw_actions
            wolp_action = actions[[i for i in range(len(actions))], 0]

        # self.action_space.tsne_render(point=raw_wolp_action,neraest_point_id=action_index)
        return raw_wolp_action, action_index

    def evaluate(
        self, observation
    ):  # When evaluating the policy, we select the action with the highest probability

        s = torch.unsqueeze(torch.tensor(observation, dtype=torch.float), 0).to(
            self.device
        )
        _, _, mean = self.actor.get_action(
            s
        )  # When choosing actions, we do not need to compute log_pi
        action = mean.cpu().detach().numpy().flatten()
        return action

    def random_action(self):

        action_id = np.random.randint(low=0, high=Action.action_space)
        action = self.action_embedding.vector_space[action_id]
        return action

    def update(self):
        batch_s, batch_a, batch_p_a, batch_r, batch_s_, batch_dw = self.memory.sample(
            self.batch_size
        )  # Sample a batch
        batch_s = batch_s.to(self.device)
        batch_a = batch_a.to(self.device)
        batch_p_a = batch_p_a.to(self.device)
        batch_s_ = batch_s_.to(self.device)
        batch_r = batch_r.to(self.device)
        batch_dw = batch_dw.to(self.device)

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(batch_s_)
            target_Q1 = self.critic_target_1(batch_s_, next_state_actions)
            target_Q2 = self.critic_target_2(batch_s_, next_state_actions)
            min_qf_next_target = (
                torch.min(target_Q1, target_Q2)
                - self.alpha.to(self.device) * next_state_log_pi
            )
            next_q_value = batch_r + (1 - batch_dw) * self.gamma * (min_qf_next_target)
        # Compute critic loss
        current_Q1 = self.critic_1(batch_s, batch_a)
        current_Q2 = self.critic_2(batch_s, batch_a)
        critic_loss = (
            F.mse_loss(current_Q1, next_q_value) + F.mse_loss(current_Q2, next_q_value)
        ) / 2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(self.critic_1.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.critic_2.parameters(), 0.5)
        self.critic_optimizer.step()

        if self.training_step % self.policy_frequency == 0:
            for _ in range(
                self.policy_frequency
            ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi, _ = self.actor.get_action(batch_s)
                Q1 = self.critic_1(batch_s, pi)
                Q2 = self.critic_2(batch_s, pi)
                Q = torch.min(Q1, Q2)
                actor_loss = ((self.alpha.to(self.device) * log_pi) - Q).mean()
                if self.use_distance_loss:

                    # batch_distance = SiameseDistanceMetric.COSINE_DISTANCE(batch_a, pi)
                    distance_loss = self.distance_loss(batch_a, pi, torch.sign(batch_r))
                    d_value = distance_loss.item()
                    actor_loss += self.distance_loss_beta * distance_loss

                else:
                    d_value = 0
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.use_grad_clip:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                if self.adaptive_alpha:
                    # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(batch_s)
                    alpha_loss = -(
                        self.log_alpha.to(self.device)
                        * (log_pi + self.target_entropy).detach()
                    ).mean()

                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    if self.use_grad_clip:
                        nn.utils.clip_grad_norm_([self.log_alpha], 0.5)
                    self.alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp()

        if self.training_step % self.target_network_frequency == 0:
            # Softly update target networks
            for param, target_param in zip(
                self.critic_1.parameters(), self.critic_target_1.parameters()
            ):
                target_param.data.copy_(
                    self.TAU * param.data + (1 - self.TAU) * target_param.data
                )
            for param, target_param in zip(
                self.critic_2.parameters(), self.critic_target_2.parameters()
            ):
                target_param.data.copy_(
                    self.TAU * param.data + (1 - self.TAU) * target_param.data
                )

    def save(self, path):
        if self.use_state_norm:
            mean = self.state_norm.running_ms.mean
            std = self.state_norm.running_ms.std
            mean_checkpoint = os.path.join(path, f"{self.name}-norm_mean.pt")
            std_checkpoint = os.path.join(path, f"{self.name}-norm_std.pt")
            torch.save(mean, mean_checkpoint)
            torch.save(std, std_checkpoint)
        actor_checkpoint = os.path.join(path, f"{self.name}-actor.pt")
        critic_checkpoint_1 = os.path.join(path, f"{self.name}-critic_1.pt")
        critic_checkpoint_2 = os.path.join(path, f"{self.name}-critic_2.pt")
        torch.save(self.actor.state_dict(), actor_checkpoint)
        torch.save(self.critic_1.state_dict(), critic_checkpoint_1)
        torch.save(self.critic_2.state_dict(), critic_checkpoint_2)

    def load(self, path):
        if self.use_state_norm:
            mean_checkpoint = os.path.join(path, f"{self.name}-norm_mean.pt")
            std_checkpoint = os.path.join(path, f"{self.name}-norm_std.pt")
            mean = torch.load(mean_checkpoint)
            std = torch.load(std_checkpoint)
            self.state_norm.running_ms.mean = mean
            self.state_norm.running_ms.std = std
        actor_checkpoint = os.path.join(path, f"{self.name}-actor.pt")
        critic_checkpoint_1 = os.path.join(path, f"{self.name}-critic_1.pt")
        critic_checkpoint_2 = os.path.join(path, f"{self.name}-critic_2.pt")
        if torch.cuda.is_available():
            self.actor.load_state_dict(torch.load(actor_checkpoint))
            self.critic_1.load_state_dict(torch.load(critic_checkpoint_1))
            self.critic_2.load_state_dict(torch.load(critic_checkpoint_2))
        else:
            self.actor.load_state_dict(
                torch.load(actor_checkpoint, map_location=torch.device("cpu"))
            )
            self.critic_1.load_state_dict(
                torch.load(critic_checkpoint_1, map_location=torch.device("cpu"))
            )
            self.critic_2.load_state_dict(
                torch.load(critic_checkpoint_2, map_location=torch.device("cpu"))
            )
        self.is_loaded_agent = True

    def train(self, target_list, eval_freq=5):
        start = time.time()
        self.num_episodes = 1
        """
        explore stage: prepare transitions
        """
        with tqdm(
            range(self.explore_eps), desc=color.color_str("Exploring", c=color.RED)
        ) as tbar:
            for _ in tbar:
                ep_results = self.run_train_episode(target_list, explore=True)
                ep_return, ep_steps, success_rate = ep_results
                tbar.set_postfix(
                    ep_return=color.color_str(f"{ep_return}", c=color.PURPLE),
                    ep_steps=color.color_str(f"{ep_steps}", c=color.GREEN),
                )
        """
        exploit stage: train policy
        
        """
        with tqdm(
            range(self.train_episodes),
            desc=f"{color.color_str('Training',c=color.RED)}",
        ) as tbar:
            for _ in tbar:
                start = time.time()
                ep_results = self.run_train_episode(target_list)
                end = time.time()
                run_time = float(end - start)

                ep_return, ep_steps, success_rate = ep_results
                self.logger.add_scalar("return-episode", ep_return, self.num_episodes)
                self.logger.add_scalar(
                    "episode-steps-episode", ep_steps, self.num_episodes
                )
                self.num_episodes += 1

                tbar.set_postfix(
                    reward=color.color_str(
                        f"{ep_return}/{self.best_return}", c=color.PURPLE
                    ),
                    step=color.color_str(f"{ep_steps}", c=color.GREEN),
                    SR=color.color_str(f"{success_rate*100}%", c=color.YELLOW),
                )

        end = time.time()
        run_time = round(end - start)
        run_time = time.strftime("%H:%M:%S", time.gmtime(run_time))
        logging.info("Training complete")
        logging.info("training time = " + run_time)

        self.logger.close()
        # for a in self.best_action_set:
        #     action = Action.get_action(a)
        #     color.print(f"[{action}]", end=" --> ")
        # print(self.best_action_set)
        # print(self.best_reward_episode)

    def run_train_episode(self, target_list, explore=False):

        steps = 0
        episode_return = 0
        self.action_set = []
        self.action_set_str = []
        self.action_set_vectors = []
        self.reward_set = []
        success_num = 0
        failed_num = 0
        target_id = 0

        random.shuffle(target_list)
        # target_list.reverse()
        while target_id < len(target_list):
            done = 0
            target_step = 0
            target: HOST = target_list[target_id]
            o = target.reset()
            if self.use_state_norm:
                # o = self.state_norm(o, update=not (self.is_loaded_agent and explore))
                o = self.state_norm(o)
            while not done and target_step < self.step_limit:
                if explore:

                    if self.is_loaded_agent:
                        proto_action = self.select_action(o)
                    else:
                        proto_action = self.random_action()
                else:
                    proto_action = self.select_action(o)
                self.action_set_vectors.append(proto_action)
                raw_wolp_action, action_index = self.action_refinement(
                    proto_action=proto_action, state=o
                )

                self.total_steps += 1
                self.action_set.append(action_index)
                self.action_set_str.append(Action.get_action(action_index))
                if 0 in self.action_set or action_index == 0:
                    self.total_action_set.add(action_index)
                next_o, r, done, result = target.perform_action(action_index)
                action_to_strore = raw_wolp_action
                if done:
                    success_num += 1
                    dw = True
                else:
                    dw = False
                if self.use_state_norm:
                    next_o = self.state_norm(next_o)

                self.memory.store(o, action_to_strore, proto_action, r, next_o, dw)
                self.reward_set.append(r)
                o = next_o.astype(np.float32)
                steps += 1
                target_step += 1
                if not explore:
                    self.training_step += 1
                    self.update()
                episode_return += r

            # if done:
            if not done:
                failed_num += 1
                if not explore:
                    break
            target_id += 1
            # if steps >= self.max_steps:
            #     break
        sucess_rate = float(format(success_num / len(target_list), ".3f"))

        if episode_return >= self.best_return:
            self.best_return = episode_return
            self.best_action_set = self.action_set
            self.best_reward_episode = self.reward_set
            self.best_episode = self.num_episodes

        return episode_return, steps, sucess_rate
