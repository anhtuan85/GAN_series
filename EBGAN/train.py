import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import pullaway_loss

def train(G, D, train_loader, latent_dim, n_epochs= 100, batch_size= 16, lr= 1e-3, optimG, optimD):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	lambda_pt= 0.1
	margin= max(1, batch_size/64)
	G.to(device)
	D.to(device)
	G.train()
	D.train()
	pixelwise_loss = nn.MSELoss()
	for epoch in range(n_epochs):
		for i, (real_images, _) in enumerate(train_loader):
			real_images = real_images.to(device)
			
			#TRain Generator 
			optimG.zero_grad()
			z = Variable(Tensor(np.random.normal(0, 1, (real_images.shape[0], latent_dim))))
			
			gen_images = G(z)
			recon_imgs, img_embeddings = D(gen_images)
			g_loss = pixelwise_loss(recon_imgs, gen_images.detach()) + lambda_pt * pullaway_loss(img_embeddings)
			
			g_loss.backward()
			optimG.step()
			
			#Train Discriminator
			optimD.zero_grad()
			real_recon, _ = D(real_images)
			fake_recon, _ = D(gen_images.detach())
			
			d_loss_real = pixelwise_loss(real_recon, real_images)
			d_loss_fake = pixelwise_loss(fake_recon, gen_images.detach())
			
			d_loss = d_loss_real
			if (margin - d_loss_fake.data).item() > 0:
				d_loss += margin - d_loss_fake
			
			d_loss.backward()
			optimD.step()
			
			print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, n_epochs, d_loss.item(), g_loss.item()))