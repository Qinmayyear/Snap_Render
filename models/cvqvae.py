import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.vq_loss import VectorQuantizer

# encoder to map input images to a latent space
class Encoder(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=128, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 4, 2, 1), # 128x128
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1), # 64x64
            nn.ReLU(),
            nn.Conv2d(hidden_channels, z_dim, 1) #z_dim x 64 x 64
        )

    def forward(self, x):
        return self.net(x)

# decoder to reconstruct images from the latent space
class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_channels=128, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_channels, 4, 2, 1), # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1), # 256x256
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
            nn.Sigmoid() 
        )

    def forward(self, z):
        return self.net(z)


class CVQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64):
        super().__init__()
        self.encoder = Encoder(z_dim=embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(z_dim=embedding_dim)

    def forward(self, x, target):
        z_e = self.encoder(x) # encode condition (foreground image)
        z_q, vq_loss = self.vq(z_e) # quantize to codebook (background image)
        x_recon = self.decoder(z_q) # decode to reconstruct image
        return x_recon, vq_loss
    

