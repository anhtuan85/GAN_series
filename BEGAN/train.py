import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model import Generator, Discriminator
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

ap.add_argument("--batch_size", default= 16, type= int, help= "Batch size for training")
ap.add_argument("--lr", "--learning-rate", default= 1e-4, type= float, help= "Learning rate")
ap.add_argument("--image_size", default= 32, type= int, help= "Size of image for training")
ap.add_argument("--h", default = 64, type= int, help= "h in BEGAN")
ap.add_argument("--n", default = 128, type = int, help = "n in BEGAN")
ap.add_argument("--beta1", default = 0.5, type= float, help = "Beta 1 in Adam optimizer")
ap.add_argument("--beta2", default = 0.999, type= float, help= "Beta 2 in Adam optimizer")
ap.add_argument("--gamma", default = 0.5, type= float, help= "Gamma in BEGAN")
ap.add_argument("--lambda_k", default = 0.001, type= float, help = "lambda k in BEGAN")
ap.add_argument("--epochs", default = 200, type= int, help = "Num epochs for training")
args = ap.parse_args()

batch_size = args.batch_size
lr = args.lr
img_size = args.image_size
h = args.h
n = args.n
beta1 = args.beta1
beta2 = args.beta2
gamma = args.gamma
lambda_k = args.lambda_k
epochs = args.epochs
k = 0.0

transform = transforms.Compose([
                                transforms.Resize(img_size), 
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])]) 
								
train_data = datasets.MNIST( "../../data/mnist", train= True, download= True, transform= transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size= batch_size, shuffle= True)

#Load generator and discriminator
G = Generator(h, n).to(device)
D = Discriminator(h, n).to(device)

#Adam optimizer
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

#Training
G.train()
D.train()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = Variable(real_images.type(Tensor))

        #Train G
        optimizerG.zero_grad()

        z = Variable(Tensor(np.random.normal(0, 1, (real_images.shape[0], h))))

        fake_images = G(z)
        g_loss = torch.mean(torch.abs(D(fake_images) - fake_images))

        g_loss.backward()
        optimizerG.step()

        #Train D
        optimizerD.zero_grad()

        d_real = D(real_images)
        d_fake = D(fake_images.detach())

        d_loss_real = torch.mean(torch.abs(d_real - real_images))
        d_loss_fake = torch.mean(torch.abs(d_fake - fake_images.detach()))

        d_loss = d_loss_real - k * d_loss_fake

        d_loss.backward()
        optimizerD.step()

        #Update
        diff = torch.mean(gamma * d_loss_real - d_loss_fake)
        k = k + lambda_k * diff.item()
        k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

        M = (d_loss_real + torch.abs(diff)).data
    
    print("[Epoch %d/%d] [D loss: %f] [G loss: %f][M: %f], k= %f" % (epoch, epochs, d_loss.item(), g_loss.item(), M, k))