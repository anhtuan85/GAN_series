import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, h, n, img_shape):
        super(Generator, self).__init__()
        self.n = n
        self.h = h

        channel, height, width = img_shape
        self.num_blocks = int(np.log2(width) - 2)
        self.fc = nn.Linear(h, 8*8*n)
        conv_layers = []

        for i in range(self.num_blocks):
            conv_layers.append(nn.Conv2d(n, n, kernel_size= 3, stride= 1, padding= 1))
            conv_layers.append(nn.ELU())
            conv_layers.append(nn.Conv2d(n,n ,kernel_size= 3, stride= 1, padding= 1))
            conv_layers.append(nn.ELU())

            if i < self.num_blocks -1 :
                conv_layers.append(nn.UpsamplingNearest2d(scale_factor= 2))

        conv_layers.append(nn.Conv2d(n, channel, kernel_size= 3, stride= 1, padding = 1))
        self.conv = nn.Sequential(*conv_layers)
    
    def forward(self, x):
        fc_out = self.fc(x).view(-1, self.n, 8, 8)
        return self.conv(fc_out)

class Discriminator(nn.Module):
    def __init__(self, h, n, img_shape):
        super(Discriminator, self).__init__()
        self.n = n
        self.h = h

        channel, height, width = img_shape
        self.num_blocks = int(np.log2(width) - 2)
        encode_layers = []
        encode_layers.append(nn.Conv2d(channel, n, kernel_size= 3, stride= 1, padding = 1))

        prev_channel_size = n

        for i in range(self.num_blocks):
            channel_size = (i+1)*n
            encode_layers.append(nn.Conv2d(prev_channel_size, channel_size, 
                                           kernel_size=3, stride= 1, padding = 1))
            encode_layers.append(nn.ELU())
            encode_layers.append(nn.Conv2d(channel_size, channel_size, 
                                           kernel_size= 3, stride= 1, padding= 1))
            encode_layers.append(nn.ELU())

            if i < self.num_blocks - 1:
                #Downsampling
                encode_layers.append(nn.Conv2d(channel_size, channel_size,
                                               kernel_size= 3, stride= 2, padding= 1))
                encode_layers.append(nn.ELU())
            
            prev_channel_size = channel_size
        self.encoder = nn.Sequential(*encode_layers)

        self.fc_encoder = nn.Linear(8*8*self.num_blocks*n, h)
        self.fc_decoder = nn.Linear(h, 8 * 8 * n)

        decode_layers = []
        for i in range(self.num_blocks):
            decode_layers.append(nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1))
            decode_layers.append(nn.ELU())
            decode_layers.append(nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1))
            decode_layers.append(nn.ELU())

            if i < self.num_blocks - 1:
                decode_layers.append(nn.UpsamplingNearest2d(scale_factor=2))
        decode_layers.append( nn.Conv2d(n,channel, kernel_size=3, stride=1, padding=1))
        self.decoder = nn.Sequential(*decode_layers)

    def forward(self, x):
        #Encoder
        x = self.encoder(x).view(x.size(0), -1)
        x = self.fc_encoder(x)

        #Decoder
        x = self.fc_decoder(x).view(-1,self.n,8,8)
        x = self.decoder(x)
        return x