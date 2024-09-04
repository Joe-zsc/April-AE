import sys, os
import math
from pprint import pprint

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
from util import Configure





class SAC_Config:
    def __init__(self):
        self.train_eps = 500  # max training episodes
        self.step_limit = 100
        self.explore_eps = 30
        # self.eval_eps = 50
        self.batch_size = 2048
        # self.memory_size = 1e5
        self.memory_size = self.train_eps * self.step_limit
        self.random_step = 1e4
        self.gamma = 0.99
        self.actor_lr = 1e-4
        self.critic_lr = 1e-4
        self.lr_alpha = 5e-5
        self.tau = 5e-2
        self.hidden_sizes = 1024
        self.eval_step_limit = 5
        self.random_radio = 0.01
        # self.target_entropy = -max(1.0, 0.98*math.log(self.action_dim))  #
        self.target_entropy = -10
        self.use_grad_clip = False
        self.adaptive_alpha = True
        self.use_state_norm = True
        self.k_nearest_neighbors = 100
        self.activate_func = "leaky_relu"
        self.action_refinement = "UCB"  # Greedy   Random
        self.use_distance_loss = "ContrastiveLoss"  
        self.distance_loss_beta = 0.1 * self.k_nearest_neighbors
        self.ucb_lamba = 1.0



