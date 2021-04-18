import torch
import random
import numpy as np
import cv2
from collections import deque

action_space = [
   [1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]
]

default_action = 4



class DQNAgent:
    def __init__(self, DQN):
        self.DQN = DQN

    def get_action(self, state, epsilon):
        self.DQN.policy_net.eval()
        with torch.no_grad():
            Qp = self.DQN.policy_net(state).cpu().numpy()
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

        init_view = np.array(game.view, copy=True) / 255
        self.init_view = [cv2.resize(init_view, dsize=sampled_img_size, interpolation=cv2.INTER_AREA).transpose(2, 0, 1)] * \
                         self.samples_per_state
        self.state = self.init_view


    def collect(self, experience):
        self.experience_replay.append(experience)

    def sample_from_experience(self, sample_size):
        if len(self.experience_replay) - 1 < sample_size:
            sample_size = len(self.experience_replay) - 1
        sample_idx = random.sample(range(len(self.experience_replay) - 1), sample_size)
        state = torch.cat([self.experience_replay[idx][0] for idx in sample_idx], dim=0).to(self.device)
        action = torch.cat([self.experience_replay[idx][1] for idx in sample_idx], dim=0).to(self.device)
        reward = torch.cat(
            [self.experience_replay[idx + 1][2] - self.experience_replay[idx][2]
            if len(self.experience_replay[idx]) == 3
            else self.experience_replay[idx][4] - self.experience_replay[idx][2]
            for idx in sample_idx], 
            dim=0
        ).to(self.device)
        next_state = torch.cat(
            [self.experience_replay[idx + 1][0]
             if len(self.experience_replay[idx]) == 3
             else self.experience_replay[idx][3]
             for idx in sample_idx],
            dim=0
        ).to(self.device)
        return state, action, reward, next_state

    def gain_experience(self, agent, num_samples, epsilon):
        scores = []
        while True:
            action = default_action
            if self.game_step % self.state_interval == 0:
                self.state = torch.tensor([self.state]).float()
                action = agent.get_action(self.state.to(self.device), epsilon)
                self.collect([self.state,  
                              torch.tensor([action]).float(), 
                              torch.tensor([self.game.get_score()]).float(),
                              ])
                self.state = []
                num_samples -= 1

            is_over = not self.game.step(action_space[action])
            if self.game_step % self.sampling_interval == 0:
                view = np.array(self.game.view, copy=True) / 255
                view = cv2.resize(view, dsize=self.sampled_img_size, interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
                self.state.append(view)

            if  is_over:
                score = self.game.get_score()
                scores.append(score)
                if len(self.state) < self.samples_per_state:
                    for _ in range(self.samples_per_state - len(self.state)):
                        self.state.append(self.state[-1])
                self.experience_replay[-1].append(torch.tensor([self.state]).float())
                self.experience_replay[-1].append(torch.tensor([score]).float())
                self.game.reset()
                self.state = self.init_view
                self.game_step = 0
            else:
                self.game_step += 1

            if num_samples == 0:
                break                
                

        return np.mean(scores) if len(scores) > 0 else 0



