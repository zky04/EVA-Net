"""
EVA-Net: Efficient Variational Alignment Network for Brain Age Prediction from EEG

Architecture:
- ProbSparse Self-Attention (O(L log L) complexity)
- Variational Information Bottleneck (VIB)
- Continuous Prototype Network
- Multi-task learning with prototype alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence data"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]


class AgeEmbedding(nn.Module):
    """Fourier/sinusoidal encoding for scalar age values"""

    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, age):
        """
        Args:
            age: Tensor of shape [batch_size] containing scalar ages
        Returns:
            age_embed: Tensor of shape [batch_size, embed_dim]
        """
        batch_size = age.size(0)
        device = age.device

        # Create frequency bands
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)

        # age shape: [batch_size, 1], emb shape: [half_dim]
        emb = age.unsqueeze(1) * emb.unsqueeze(0)  # [batch_size, half_dim]

        # Concatenate sin and cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [batch_size, embed_dim]

        return emb


class ProbSparseSelfAttention(nn.Module):
    """
    ProbSparse Self-Attention mechanism with O(L log L) complexity.
    Selects top-u queries based on sparsity measurement (KL divergence approximation).
    """

    def __init__(self, d_model, n_heads, sampling_factor=5):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.sampling_factor = sampling_factor

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            out: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections and reshape for multi-head
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # ProbSparse attention: select top-u queries
        u = int(self.sampling_factor * math.log(seq_len))
        u = max(1, min(u, seq_len))  # Ensure u is valid

        # Compute sparsity measurement: M(q_i, K) for each query
        # M(q_i, K) = max_j(q_i·k_j / sqrt(d_k)) - mean_j(q_i·k_j / sqrt(d_k))
        Q_K = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, L, L]

        # Max over keys for each query
        M_max = Q_K.max(dim=-1)[0]  # [B, H, L]
        # Mean over keys for each query
        M_mean = Q_K.mean(dim=-1)  # [B, H, L]
        # Sparsity score
        M = M_max - M_mean  # [B, H, L]

        # Select top-u queries
        _, top_indices = torch.topk(M, u, dim=-1)  # [B, H, u]

        # Gather top queries
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_k)  # [B, H, u, d_k]
        Q_reduced = torch.gather(Q, 2, top_indices_expanded)  # [B, H, u, d_k]

        # Compute attention for top queries
        scores = torch.matmul(Q_reduced, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, u, L]
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # [B, H, u, d_k]

        # Create output tensor and scatter back
        out = torch.zeros_like(Q)  # [B, H, L, d_k]
        out.scatter_(2, top_indices_expanded, context)

        # For non-selected queries, use mean of values
        V_mean = V.mean(dim=2, keepdim=True)  # [B, H, 1, d_k]
        mask = torch.ones(batch_size, self.n_heads, seq_len, 1, device=x.device)
        mask.scatter_(2, top_indices.unsqueeze(-1), 0)
        out = out + mask * V_mean

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.fc_out(out)

        return out


class InformerEncoderLayer(nn.Module):
    """Single Informer encoder layer with ProbSparse attention"""

    def __init__(self, d_model, n_heads, d_ff=512, dropout=0.1, sampling_factor=5):
        super().__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, n_heads, sampling_factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class EfficientEncoder(nn.Module):
    """
    Efficient long-sequence encoder with ProbSparse attention.
    Processes EEG input: [batch_size, 19, 1000] -> hidden state [batch_size, d_model]
    """

    def __init__(self, n_channels=19, d_model=128, n_layers=4, n_heads=8,
                 d_ff=512, dropout=0.1, sampling_factor=5):
        super().__init__()

        # Channel-wise projection to d_model
        self.input_projection = nn.Linear(n_channels, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=1000)

        # Stack of Informer encoder layers
        self.layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout, sampling_factor)
            for _ in range(n_layers)
        ])

        self.d_model = d_model

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_channels, seq_len] = [B, 19, 1000]
        Returns:
            h: [batch_size, d_model]
        """
        # Transpose to [batch_size, seq_len, n_channels]
        x = x.transpose(1, 2)  # [B, 1000, 19]

        # Project channels to d_model
        x = self.input_projection(x)  # [B, 1000, d_model]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x)

        # Global average pooling over time
        h = x.mean(dim=1)  # [B, d_model]

        return h


class VariationalInformationBottleneck(nn.Module):
    """
    VIB module: maps hidden state H to latent code Z via reparameterization trick.
    Computes KL divergence loss for information bottleneck regularization.
    """

    def __init__(self, d_model=128, latent_dim=64):
        super().__init__()
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, h):
        """
        Args:
            h: [batch_size, d_model] - hidden representation
        Returns:
            z: [batch_size, latent_dim] - sampled latent code
            kl_loss: scalar - KL divergence for regularization
        """
        mu = self.fc_mu(h)  # [B, latent_dim]
        logvar = self.fc_logvar(h)  # [B, latent_dim]

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        # KL divergence: D_KL(N(mu, sigma^2) || N(0, I))
        # = 0.5 * sum(sigma^2 + mu^2 - log(sigma^2) - 1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = kl_loss.mean()  # Average over batch

        return z, kl_loss


class ContinuousPrototypeNetwork(nn.Module):
    """
    Maps scalar age to ideal prototype in latent space.
    P_theta: R -> R^d (age -> prototype embedding)
    """

    def __init__(self, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.age_embedding = AgeEmbedding(embed_dim=128)
        self.mlp = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, age):
        """
        Args:
            age: [batch_size] - scalar ages
        Returns:
            prototype: [batch_size, latent_dim] - ideal age-conditioned prototypes
        """
        age_embed = self.age_embedding(age)  # [B, 128]
        prototype = self.mlp(age_embed)  # [B, latent_dim]
        return prototype


class PredictionHead(nn.Module):
    """MLP that maps latent code Z to predicted age"""

    def __init__(self, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        """
        Args:
            z: [batch_size, latent_dim]
        Returns:
            age_pred: [batch_size, 1]
        """
        return self.mlp(z)


class EVANet(nn.Module):
    """
    Complete EVA-Net framework:
    - Efficient encoder (ProbSparse attention)
    - Variational Information Bottleneck
    - Continuous Prototype Network
    - Prediction Head
    """

    def __init__(self,
                 n_channels=19,
                 d_model=128,
                 n_layers=4,
                 n_heads=8,
                 d_ff=512,
                 dropout=0.1,
                 sampling_factor=5,
                 latent_dim=64,
                 hidden_dim=256):
        super().__init__()

        self.encoder = EfficientEncoder(
            n_channels=n_channels,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            sampling_factor=sampling_factor
        )

        self.vib = VariationalInformationBottleneck(d_model, latent_dim)
        self.prototype_net = ContinuousPrototypeNetwork(latent_dim, hidden_dim)
        self.prediction_head = PredictionHead(latent_dim, hidden_dim)

    def forward(self, x, age):
        """
        Args:
            x: [batch_size, 19, 1000] - EEG input
            age: [batch_size] - chronological ages
        Returns:
            age_pred: [batch_size, 1] - predicted ages
            z: [batch_size, latent_dim] - latent representations
            prototype: [batch_size, latent_dim] - age-conditioned prototypes
            kl_loss: scalar - VIB regularization loss
        """
        # Encode input to hidden state
        h = self.encoder(x)  # [B, d_model]

        # VIB: map to latent space
        z, kl_loss = self.vib(h)  # [B, latent_dim], scalar

        # Generate age-conditioned prototype
        prototype = self.prototype_net(age)  # [B, latent_dim]

        # Predict age from latent code
        age_pred = self.prediction_head(z)  # [B, 1]

        return age_pred, z, prototype, kl_loss

    def compute_prototype_alignment_error(self, z, prototype):
        """
        Compute PAE: ||z - P_y||_2
        Args:
            z: [batch_size, latent_dim]
            prototype: [batch_size, latent_dim]
        Returns:
            pae: [batch_size] - alignment error for each sample
        """
        return torch.norm(z - prototype, p=2, dim=1)
