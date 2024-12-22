
class April_AE_Config:
    #SAC version
    def __init__(self):
        self.train_eps = 500  # max training episodes
        self.step_limit = 100
        self.explore_eps = 30
        self.batch_size = 2048
        # self.memory_size = 1e5
        self.memory_size = self.train_eps * self.step_limit
        self.gamma = 0.9
        self.actor_lr = 1e-4
        self.critic_lr = 1e-4
        self.lr_alpha = 5e-5
        self.tau = 5e-2
        self.hidden_sizes = 1024
        self.eval_step_limit = 5
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



