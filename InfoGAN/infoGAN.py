import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable

#Hyperparameters
batch_size = 64
num_workers = 0
num_epochs = 150
dis_c_dim = 10
num_con_c= 2
z_dim = 74
num_z = 62
num_dis_c= 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([ 
                                transforms.Resize(28),
                                transforms.CenterCrop(28),
                                transforms.ToTensor()])
train_data = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size= batch_size, num_workers= num_workers)

def noise_sample(n_dis_c, dis_c_dim, n_con_c, n_z, batch_size):
    z = torch.randn(batch_size, n_z, 1, 1, device= device)

    idx = np.zeros((n_dis_c, batch_size))
    if(n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device= device)
        
        for i in range(n_dis_c):
            idx[i] = np.random.randint(dis_c_dim, size=batch_size)
            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if(n_con_c != 0):
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device= device) * 2 - 1

    noise = z
    if(n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if(n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.tconv1 = nn.ConvTranspose2d(74, 1024, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)

        self.tconv2 = nn.ConvTranspose2d(1024, 128, 7, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))

        img = torch.sigmoid(self.tconv4(x))

        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 1024, 7, bias=False)
        self.bn3 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
        return x

class DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1024, 1, 1)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))

        return output

class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1024, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(128, 10, 1)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var

G = Generator().to(device)
D = Discriminator().to(device)
D_Head = DHead().to(device)
Q_Head = QHead().to(device)

def real_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.ones(batch_size).to(device)
    criterion = nn.BCELoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size).to(device) # fake labels = 0
    criterion = nn.BCELoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, x, mu, var):
        
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll

d_optimizer = optim.Adam([{'params': D.parameters()}, {'params': D_Head.parameters()}], lr= 0.0002)
g_optimizer = optim.Adam([{'params': G.parameters()}, {'params': Q_Head.parameters()}], lr= 0.001)


real_label = 1
fake_label = 0
G.train()
D.train()
D_Head.train()
Q_Head.train()
criterionD = nn.BCELoss()
Q_discre_criterion = nn.CrossEntropyLoss()
Q_con_criterion = NormalNLLLoss()
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        #Train Discriminator
        d_optimizer.zero_grad()
        label = torch.full((batch_size, ), real_label, device=device)

        ouput1 = D(real_images)
        probs_real = D_Head(ouput1).view(-1)
        d_real_loss = criterionD(probs_real, label)

        d_real_loss.backward()

        #fake data
        label.fill_(fake_label)
        noise, idx = noise_sample(num_dis_c, dis_c_dim ,num_con_c, num_z, batch_size)
        fake_image = G(noise.to(device))
        output2 = D(fake_image.detach())
        probs_fake = D_Head(output2).view(-1)
        d_fake_loss = criterionD(probs_fake, label)
        d_fake_loss.backward()

        #Combine loss
        d_loss = d_real_loss + d_fake_loss
        d_optimizer.step()

        #Train generator
        output = D(fake_image)
        label.fill_(real_label)
        probs_fake = D_Head(output).view(-1)
        gen_loss = criterionD(probs_fake, label)
        
        q_logits, q_mu, q_var = Q_Head(output)
        target = torch.LongTensor(idx).to(device)
        dis_loss = 0
        for j in range(num_dis_c):
            dis_loss += Q_discre_criterion(q_logits[:, j*10 : j*10 + 10], target[j])

        con_loss = 0
        if (num_con_c != 0):
            con_loss = Q_con_criterion(noise[:, num_z+num_dis_c*dis_c_dim:].view(-1,num_con_c), q_mu, q_var)

        G_loss = gen_loss + dis_loss + con_loss
        G_loss.backward()
        g_optimizer.step()

    print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
        epoch+1, num_epochs, d_loss.item(), G_loss.item()))	