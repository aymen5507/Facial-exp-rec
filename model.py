import torch
from parameters import NETWORK, HYPERPARAMS

import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.optimizer = HYPERPARAMS.optimizer
        self.optimizer_param = HYPERPARAMS.optimizer_param
        self.learning_rate = HYPERPARAMS.learning_rate
        self.keep_prob = HYPERPARAMS.keep_prob
        self.learning_rate_decay = HYPERPARAMS.learning_rate_decay
        self.decay_step = HYPERPARAMS.decay_step

        if NETWORK.model == 'A':
            self.network = self.build_modelA()
        elif NETWORK.model == 'B':
            self.network = self.build_modelB()
        else:
            print("ERROR: no model " + str(NETWORK.model))
            exit()

    def forward(self, x):
        return self.network(x)

    def build_modelB(self):
        model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(self.keep_prob),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, NETWORK.output_size),
            nn.Softmax(dim=1)
        )
        return model

    def build_modelA(self):
        model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Dropout(self.keep_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, NETWORK.output_size),
            nn.Softmax(dim=1)
        )
        return model

model = Model()
optimizer = optim.SGD(model.parameters(), lr=model.learning_rate, momentum=model.optimizer_param)
criterion = nn.CrossEntropyLoss()
