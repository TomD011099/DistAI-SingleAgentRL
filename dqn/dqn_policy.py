from collections import deque

import torch
import torch.nn as nn
import numpy as np
import random
from ray.rllib.policy import Policy
from ray.rllib.models import ModelCatalog


class DQNPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

        # The possible observations:
        # * cart pos (-2.4, 2.4)
        # * cart vel (-Inf, Inf)
        # * pole ang (-41.8°, 41.8°)
        # * pole rot (-Inf, Inf)
        self.observation_space = observation_space

        # The possible actions:
        # * 0 - Push left
        # * 1 - Push right
        self.action_space = action_space

        # The config
        self.config = config

        # Learning rate
        self.lr = self.config["lr"]

        # Queue length
        self.ex_buf_len = self.config["ex_buf_len"]

        # Discount
        self.discount = self.config["disc"]

        # Epsilon for greedy
        self.eps = self.config["eps"]

        # Experience buffer sample size
        self.ex_buf_sample_size = self.config["ex_buf_sample_size"]

        # GPU settings
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.dqn_model = ModelCatalog.get_model_v2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=2,
            name="DQNModel",
            model_config=self.config["dqn_model"],
            framework="torch",
        ).to(self.device, non_blocking=True)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.dqn_model.parameters(), lr=self.lr)

        # Experience buffer
        # Used to replay old experiences to refresh them over time (helps with local max)
        # Entries will be tuple of obs, action, reward, done, n_obs
        self.ex_buf = deque(maxlen=self.ex_buf_len)

    def greedy(self, value):
        chance = random.randint(1, 100)
        if chance < self.eps:
            out = self.action_space.sample()
        else:
            out = torch.argmax(value).item()
        self.eps = self.eps * 0.9999
        print(self.eps)
        return out

    # Worker function
    # Will decide which action to take, based on the obs
    # Only 1 obs and subsequently, 1 action
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        explore=None,
                        timestep=None,
                        **kwargs):
        # A batch of observational data in a tensor
        obs_batch_t = torch.tensor(obs_batch).type(torch.FloatTensor)

        # What the nn returns as val, decide the action based on this
        value_batch_t = self.dqn_model(obs_batch_t)

        # The actions for the elements of the batch
        out = [self.greedy(value) for value in value_batch_t], [], {}
        return out

    # Trainer function
    # Gets the results of the workers and uses them to calculate the Q-function
    # Samples is a dict of:
    # * t               - How far into training
    # * eps_id          - Episode id (same for one t batch)
    # * agent_index     - Agent index (All 0, probably because there is only one worker)
    # * obs             - Observation (state) [s]
    # * actions         - Action decided by the worker
    # * rewards         - Reward [R(s)]
    # * prev_actions    - Prev action made to get to this state
    # * prev_rewards    - Reward from prev state
    # * dones           - Done training
    # * infos           - ???
    # * new_obs         - The state ofter doing action [s']
    def learn_on_batch(self, samples):
        # Save experiences
        for i in range(len(samples["dones"])):
            s_obs = samples["obs"][i]
            s_action = samples["actions"][i]
            s_reward = samples["rewards"][i]
            s_done = samples["dones"][i]
            s_n_obs = samples["new_obs"][i]
            self.ex_buf.append((s_obs, s_action, s_reward, s_done, s_n_obs))

        # Take samples from the experience buffer
        l_obs, l_actions, l_rewards, l_dones, l_n_obs = zip(*random.sample(self.ex_buf, self.ex_buf_sample_size))

        obs_batch_t = torch.tensor(l_obs).type(torch.FloatTensor)
        action_batch_t = torch.tensor(l_actions)
        rewards_batch_t = torch.tensor(l_rewards)
        done_t = torch.tensor(l_dones)
        new_obs_batch_t = torch.tensor(l_n_obs).type(torch.FloatTensor)
        optimal_q_t, _ = torch.max(self.dqn_model(new_obs_batch_t), dim=1)
        optimal_q_t[done_t] = 0

        action_batch_t = action_batch_t.unsqueeze(-1)
        guess_t = torch.gather(self.dqn_model(obs_batch_t), 0, action_batch_t)

        criterion = nn.MSELoss()
        # Calculate the loss
        loss = criterion(guess_t, (rewards_batch_t + self.discount * optimal_q_t))
        self.optimizer.zero_grad()
        # Backward the mean of the losses (sum() could also be used, but is bad if batch size changes)
        loss.mean().backward()
        self.optimizer.step()

        return {"learner_stats": {"loss": loss.item()}}

    # Trainer function
    def get_weights(self):
        weights = {}
        weights["dqn_model"] = self.dqn_model.cpu().state_dict()
        self.dqn_model.to(self.device, non_blocking=False)
        return weights

    # Worker function
    def set_weights(self, weights):
        if "dqn_model" in weights:
            self.dqn_model.load_state_dict(weights["dqn_model"], strict=True)
            self.dqn_model.to(self.device, non_blocking=False)
