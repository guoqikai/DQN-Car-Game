import torch
import numpy as np
from game import AutoDriveGame, Car
from util import DQNAgent, Experience, action_space
from DQN import AutoDriveDQN, QNet

if __name__ == "__main__":
    device = torch.device('cpu')

    # initialize the game
    car = Car(0, (60, 60), (17, 29), (255, 125, 0), 1, 5, 30, 0.01)
    map_ = np.zeros((640, 640, 3))

    game = AutoDriveGame(map_, car, (500, 500), 1, max_steps=512)

    # initialize the DQA and agent
    q_net = QNet([3, 4, 8, 16], [256, 512, 64, len(action_space)]).to(device)
    dqn = AutoDriveDQN(q_net, 1, 5, torch.nn.MSELoss())
    agent = DQNAgent(dqn)

    epsilon = 1
    exp_replay_size = 64
    sampling_interval = 4
    state_interval = 16
    img_sz = (128, 128)
    memory = Experience(game, exp_replay_size, sampling_interval, state_interval, img_sz, device)
    memory.gain_experience(agent, exp_replay_size, epsilon)
    episodes = 9000

    sample_size = 64
    batch_size = 32
    num_epochs = 4

    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-5)
    for i in range(episodes):
        avg_score = memory.gain_experience(agent, sample_size, epsilon)
        print(avg_score)
        losses = []
        for j in range(num_epochs * sample_size//batch_size):
            loss = dqn.loss(memory, batch_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dqn.step()
            losses.append(loss.item())

        print("avg score:",  avg_score, "avg loss:", np.mean(losses), epsilon)
        epsilon = 0.99 * epsilon
