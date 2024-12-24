import os
from RL_Model.config import April_AE_Config
from host import StateEncoder, HOST
from RL_Model.common import Normalization
from util import color
import logging
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random
from actions.Action import Action
import numpy as np
import torch


class Agent:
    def __init__(self, name, config=None):
        self.name = name
        self.config = config

        if self.name == "April-AE":
            from RL_Model.April_AE import April_AE

            if not self.config:
                self.config = April_AE_Config()
            self.Policy = April_AE(cfg=self.config)

        else:
            self.Policy = None
            print("please imput agent name")
            exit(0)
        self.is_loaded_agent = False
        self.use_state_norm = self.config.use_state_norm
        if self.use_state_norm:
            self.state_norm = Normalization(shape=StateEncoder.state_space)
        self.num_episodes = 0
        self.eval_times = 0
        self.task_num_episodes = 0
        self.total_training_step = 0
        self.best_return = -float("inf")
        self.best_action_set = []
        self.best_episode = 0
        self.best_reward_episode = []
        self.eval_rewards = 0
        self.eval_success_rate = 0
        self.mean_exp_coverage = 0
        self.last_episode_reward = -float("inf")
        self.first_hit_step = -1
        self.logger = SummaryWriter()
        print(f"--Running {self.Policy.name} agent--")

    def train(self, target_list, eval_freq=5):
        train_start = time.time()
        self.num_episodes = 0
        """
        explore stage: prepare transitions
        """
        with tqdm(
            range(self.config.explore_eps),
            desc=color.color_str("Exploring", c=color.RED),
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
            range(self.config.train_eps),
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
                # self.Policy.num_episodes += 1
                self.num_episodes += 1
                tbar.set_postfix(
                    reward=color.color_str(
                        f"{ep_return}/{self.best_return}", c=color.PURPLE
                    ),
                    step=color.color_str(f"{ep_steps}", c=color.GREEN),
                    SR=color.color_str(f"{success_rate*100}%", c=color.YELLOW),
                )

        train_end = time.time()
        run_time = round(train_end - train_start)
        run_time = time.strftime("%H:%M:%S", time.gmtime(run_time))
        logging.info("Training complete")
        logging.info("training time = " + run_time)

        self.logger.close()


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
            while not done and target_step < self.config.step_limit:

                action_info = self.Policy.select_action(
                    observation=o,
                    explore=explore,
                    is_loaded_agent=self.is_loaded_agent,
                    num_episode=self.num_episodes,
                )
                action_index = action_info[0]
                self.total_training_step += 1
                self.action_set.append(action_index)
                self.action_set_str.append(Action.get_action(action_index))
                # if 0 in self.action_set or action_index == 0:
                #     self.total_action_set.add(action_index)
                next_o, r, done, result = target.perform_action(action_index)

                if done:
                    success_num += 1
                    dw = True
                else:
                    dw = False
                if self.use_state_norm:
                    next_o = self.state_norm(next_o)
                self.Policy.store_transtion(o, action_info, r, next_o, dw)
                # self.memory.store(o, action_to_strore, proto_action, r, next_o, dw)
                self.reward_set.append(r)
                o = next_o.astype(np.float32)
                steps += 1
                target_step += 1
                if not explore:
                    self.total_training_step += 1
                    self.Policy.update_policy(
                        num_episode=self.num_episodes,
                        train_steps=self.total_training_step,
                    )
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

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        assert os.path.exists(path), f"{path} does not exist"
        if self.use_state_norm:
            mean = self.state_norm.running_ms.mean
            std = self.state_norm.running_ms.std
            mean_checkpoint = path / f"state_norm_mean.pt"
            std_checkpoint = path / f"state_norm_std.pt"
            torch.save(mean, mean_checkpoint)
            torch.save(std, std_checkpoint)
        self.Policy.save(path)

    def load(self, path):
        if self.use_state_norm:
            self.state_norm = Normalization(
                shape=StateEncoder.state_space, finetune=True
            )
            mean_checkpoint = path / f"state_norm_mean.pt"
            std_checkpoint = path / f"state_norm_std.pt"
            mean = torch.load(mean_checkpoint)
            std = torch.load(std_checkpoint)
            self.state_norm.running_ms.mean = mean
            self.state_norm.running_ms.std = std
            self.state_norm.running_ms.S = std * std
        self.Policy.load(path)
        self.is_loaded_agent = True
