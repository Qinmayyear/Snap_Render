# models/cvqvae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.vq_loss import VectorQuantizer

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, 2, 1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(hidden_channels, z_dim, 1)  # Output: z_dim x 64 x 64
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_channels=128, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_channels, 4, 2, 1),  # 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1),  # 256x256
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
            nn.Sigmoid()  # Output range: [0, 1]
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
        z_e = self.encoder(x)                # Encode condition (foreground image)
        z_q, vq_loss = self.vq(z_e)          # Quantize
        x_recon = self.decoder(z_q)          # Decode to full (ideally background-only)
        return x_recon, vq_loss
    

