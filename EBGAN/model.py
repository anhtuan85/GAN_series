import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_dim= 100, channels = 1, image_size = 32):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.channels = channels
        self.image_size = image_size
        
        self.fc = nn.Sequential(
                nn.Linear(self.input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 128*(self.image_size // 4) * (self.image_size // 4)),
                nn.BatchNorm1d(128*(self.image_size // 4) * (self.image_size // 4)),
                nn.ReLU())
        self.deconv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, self.channels, 4, 2, 1),
                nn.Tanh())
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, (self.image_size // 4), (self.image_size // 4))
        x = self.deconv(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, input_size = 32):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size

        self.conv = nn.Sequential(nn.Conv2d(self.in_channels, 64, 4, 2, 1),
                                  nn.ReLU())
        self.embedding = nn.Sequential(
                nn.Linear(64 * (self.input_size // 2) * (self.input_size // 2), 32))
        
        self.fc = nn.Sequential(
                nn.Linear(32, 64 * (self.input_size //2) * (self.input_size // 2)),
                nn.BatchNorm1d(64* (self.input_size//2) * (self.input_size // 2)),
                nn.ReLU())
        
        self.deconv = nn.Sequential(nn.ConvTranspose2d(64, self.out_channels, 4, 2, 1))
    
    def forward(self, image):
        x = self.conv(image)
        x = x.view(x.size()[0], -1)
        code = self.embedding(x)
        
        x = self.fc(code)        
        x = x.view(-1, 64, (self.input_size // 2), (self.input_size // 2))
        x = self.deconv(x)
        
        return x, code