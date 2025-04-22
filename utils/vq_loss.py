# utils/vq_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z):
        # input shape: (B, C, H, W) → flatten to (BHW, C)
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        flat_z = z_perm.view(-1, self.embedding_dim)  # (B*H*W, C)

        # Compute distances to codebook
        distances = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1)
        )  # out: (BHW, num_embeddings)

        # check for nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1)
        # Convert to one-hot encoding
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()


        quantized = encodings @ self.embedding.weight  # (BHW, C)
        quantized = quantized.view(z_perm.shape)  # (B, H, W, C)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # Losses
        commitment_loss = F.mse_loss(quantized.detach(), z)
        embedding_loss = F.mse_loss(quantized, z.detach())
        loss = embedding_loss + self.beta * commitment_loss

        # Straight-through gradient
        quantized = z + (quantized - z).detach()

        return quantized, loss