import torch
import numpy as np
import pygame
from game import AutoDriveGame, Car
from util import DQNAgent, Experience, action_space
from DQN import AutoDriveDQN, QNet


if __name__ == "__main__":
    device = torch.device('cuda:0')
    show_game = True

    # initialize the game
    car = Car(0, (100, 100), (33, 43), (255, 125, 0), 4, 7, 14, 0.01)
    map_ = np.zeros((640, 640, 3))

    game = AutoDriveGame(map_, car, (500, 500), max_steps=512)

    # initialize the DQA and agent
    q_net = QNet([(4, 16, 8, 4), (16, 32, 4, 4), (32, 32, 3, 2)], [800, 256, len(action_space)]).to(device)
    dqn = AutoDriveDQN(q_net, 0.75, 100, torch.nn.MSELoss())
    agent = DQNAgent(dqn)

    epsilon = 0.5
    exp_replay_size = 4096
    sampling_interval = 1
    sample_size = 128
    state_interval = 4
    img_sz = (180, 180)
    memory = Experience(game, exp_replay_size, sampling_interval, state_interval, img_sz, device)
    episodes = 100

    if show_game:
        pygame.init()
        display = pygame.display.set_mode(img_sz)
    batch_size = 128
    num_epochs = 64

    optimizer = torch.optim.Adam(dqn.policy_net.parameters(), 1e-3)
    for i in range(episodes):
        
        avg_score = memory.gain_experience(agent, sample_size, epsilon)
       
        losses = []
        for j in range(num_epochs):
            loss = dqn.loss(memory, batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dqn.step()
            losses.append(loss.item())

        print("avg score:",  avg_score,  "avg loss:", np.mean(losses), "episodes:", i)
        epsilon = max(0.98 * epsilon, 0.05)
        if not show_game:
            continue

        for i in range(sample_size):
            for im in memory.experience_replay[len(memory.experience_replay) - sample_size + i][0][0].cpu().numpy().transpose(0, 2, 3, 1):
                surf = pygame.surfarray.make_surface(im * 255)
                display.blit(surf, (0, 0))
                pygame.display.update()
                pygame.time.wait(10)
    torch.save(dqn.policy_net.state_dict(), "./policy")
    if show_game:
        pygame.quit()


