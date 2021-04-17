import torch
import copy
import torch.nn as nn


class AutoDriveDQN:

    def __init__(self, Q_net, gamma, update_interval, loss_fn):
        self.policy_net = Q_net
        self.target_net = copy.deepcopy(Q_net)
        self.gamma = torch.tensor(gamma).float()
        self.num_step = 0
        self.update_interval = update_interval
        self.loss_fn = loss_fn

    def step(self):
        self.num_step += 1
        if self.num_step % self.update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def loss(self, memory, batch_size):
        self.policy_net.train()
        self.target_net.eval()
        state, action, reward, next_state = memory.sample_from_experience(sample_size=batch_size)
        expected = self.policy_net(state)[torch.arange(action.shape[0]), action.long()]
        target, _ = torch.max(self.target_net(next_state).detach(), axis=1)
        loss = self.loss_fn(expected.unsqueeze(1), (reward + self.gamma * target).unsqueeze(1))
        return loss


class QNet(nn.Module):
    def __init__(self, conv_layer_channels, linear_layer_sizes, batch_norm=True):
        super().__init__()

        cnn_layers = []
        for i in range(1, len(conv_layer_channels)):
            cnn_layers.append(nn.Conv2d(conv_layer_channels[i - 1], conv_layer_channels[i], 3, padding=1))
            cnn_layers.append(nn.MaxPool2d(4))
            if batch_norm:
                cnn_layers.append(nn.BatchNorm2d(conv_layer_channels[i]))
            cnn_layers.append(nn.ReLU())

        layers = []
        for i in range(1, len(linear_layer_sizes)):
            layers.append(nn.Linear(linear_layer_sizes[i - 1], linear_layer_sizes[i]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(linear_layer_sizes[i]))
            if i < len(linear_layer_sizes) - 1:
                layers.append(nn.Tanh())

        self.cnn = nn.Sequential(*cnn_layers)
        self.dnn = nn.Sequential(*layers)

    """
    inputs is batch_size * img_per_state * C * H * W
    """
    def forward(self, inputs):
        if len(inputs.shape) == 4:
            inputs = inputs.unsqueeze(0)
        B, _, H, W, C = inputs.shape
        return self.dnn(self.cnn(inputs.permute(0, 1, 4, 2, 3).view(-1, C, H, W)).view(B, -1))


