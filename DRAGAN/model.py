import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim, h_dim, y_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim

        self.fc = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.Sigmoid(),
            nn.Linear(self.h_dim, self.y_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, z):
        return self.fc(z)

class Discriminator(nn.Module):
    def __init__(self, h_dim, y_dim):
        super(Discriminator, self).__init__()
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.fc = nn.Sequential(
            nn.Linear(self.y_dim, self.h_dim),
            nn.Sigmoid(),
            nn.Linear(self.h_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, y):
        return self.fc(y)
