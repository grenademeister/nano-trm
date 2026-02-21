import torch
import torch.nn as nn

from .part import TinyNetworkBlock, TokenGate


class TRM(nn.Module):
    def __init__(self, embed_dim=512, n_reasoning_steps=6, n_recursion_steps=3, att='self_att'):
        super().__init__()
        self.net = TinyNetworkBlock(embed_dim, att=att)

        self.n = n_reasoning_steps
        self.T = n_recursion_steps
        self.token_embedding = nn.Embedding(10, embed_dim)
        self.output_head = nn.Linear(embed_dim, 10)
        self.x_gate = TokenGate(embed_dim)
        self.y_gate = TokenGate(embed_dim)

    def embed_input(self, x):
        return self.token_embedding(x)

    def latent_recursion(self, x, y, z):
        for _ in range(self.n):
            x_ = self.x_gate(y + z) * x
            y_ = self.y_gate(z) * y
            z = self.net(query=x_ + y_ + z)
        y = self.net(query=z + y)
        return y, z

    def forward(self, x, y, z):
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self.latent_recursion(x, y, z)
        y, z = self.latent_recursion(x, y, z)
        return (y.detach(), z.detach()), self.output_head(y)
