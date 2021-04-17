import torch
import random
import numpy as np
import cv2
from collections import deque

action_space = [
    [-1, -1],
    [0, -1],
    [1, -1],
    [-1, 0],
    [0, 0],
    [1, 0],
    [-1, 1],
    [0, 1],
    [1, 1]
]


class DQNAgent:
    def __init__(self, DQN):
        self.DQN = DQN

    def get_action(self, state, epsilon):
        self.DQN.policy_net.eval()
        with torch.no_grad():
            Qp = self.DQN.policy_net(torch.tensor(state).float()).numpy()
        action = np.argmax(Qp.reshape(-1))
        if random.uniform(0, 1) < epsilon:
            action = random.choice(range(len(action_space)))
        return action


class Experience:
    def __init__(self, game, size, sampling_interval, state_interval, sampled_img_size, device):
        assert state_interval % sampling_interval == 0
        self.experience_replay = deque(maxlen=size)
        self.samples_per_state = state_interval // sampling_interval
        self.state_interval = state_interval
        self.sampling_interval = sampling_interval
        self.sampled_img_size = sampled_img_size
        self.device = device
        game.reset()
        self.game = game
        self.game_step = 0

        init_view = np.array(game.view, copy=True)
        self.init_view = [cv2.resize(init_view, dsize=sampled_img_size, interpolation=cv2.INTER_CUBIC)] * \
                         self.samples_per_state
        self.previous_state = self.init_view
        self.previous_score = 0


    def collect(self, experience):
        self.experience_replay.append(experience)

    def sample_from_experience(self, sample_size):
        if len(self.experience_replay) < sample_size:
            sample_size = len(self.experience_replay)
        sample_idx = random.sample(range(len(self.experience_replay)), sample_size)
        state = torch.tensor([self.experience_replay[idx][0] for idx in sample_idx], device=self.device).float()
        action = torch.tensor([self.experience_replay[idx][2] for idx in sample_idx], device=self.device).float()
        reward = torch.tensor([self.experience_replay[idx][3] for idx in sample_idx], device=self.device).float()
        next_state = torch.tensor(
            [self.experience_replay[idx + 1][0]
             if len(self.experience_replay) > idx + 1 and self.experience_replay[idx + 1][1] != 0
             else self.experience_replay[idx][0]
             for idx in sample_idx],
            device=self.device
        ).float()
        return state, action, reward, next_state

    def gain_experience(self, agent, num_samples, epsilon):
        state = []
        scores = []
        while True:
            action = 4
            if self.game_step % self.state_interval == 0:
                action = agent.get_action(self.previous_state, epsilon)
            is_over = not self.game.step(action_space[action])
            if self.game_step % self.sampling_interval == 0 or is_over:
                view = np.array(self.game.view, copy=True)
                view = cv2.resize(view, dsize=self.sampled_img_size)
                state.append(view)
            if self.game_step % self.state_interval == 0 or is_over:
                score = self.game.get_score()
                if len(state) < self.samples_per_state:
                    for _ in range(self.samples_per_state - len(state)):
                        state.append(state[-1])
                self.collect([state, self.game_step, action, score - self.previous_score])
                self.previous_state = state
                self.previous_score = score
                state = []
                num_samples -= 1
                if num_samples == 0:
                    break
            self.game_step += 1

            if is_over:
                scores.append(self.game.get_score())
                self.game.reset()
                self.game_step = 0
                self.previous_state = self.init_view
                self.previous_score = 0

        return np.mean(scores) if len(scores) > 0 else 0



