import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.init import xavier_normal

def visualize(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    
    #get one image in batch
    img = np.squeeze(images[5])
    fig = plt.figure(figsize = (3,3)) 
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')

def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=10, ncols=10, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((32,32)).to("cpu"), cmap='Greys_r')

def xavier_init(model):
    for param in model.parameters():
        if len(param.size()) ==2:
            xavier_normal(param)

