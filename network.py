import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TextEmbedding(nn.Module):
    """Simple embedding layer for digit labels (0-9)."""
    def __init__(self, num_classes=10, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
    
    def forward(self, labels):
        return self.embedding(labels)


class Block(nn.Module):
    """Basic residual block with group normalization."""
    def __init__(self, channels_in, channels_out, time_embedding_dim, text_embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels_out)
        self.norm2 = nn.GroupNorm(8, channels_out)
        
        # Time embedding projection
        self.time_mlp = nn.Linear(time_embedding_dim, channels_out)
        
        # Text embedding projection
        self.text_mlp = nn.Linear(text_embedding_dim, channels_out)
        
        # Residual connection
        self.residual_conv = nn.Conv2d(channels_in, channels_out, kernel_size=1) if channels_in != channels_out else nn.Identity()
    
    def forward(self, x, time_emb, text_emb):
        residual = self.residual_conv(x)
        
        # First conv
        x = self.conv1(x)
        x = self.norm1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        x = x + time_emb[:, :, None, None]
        
        # Add text embedding
        text_emb = self.text_mlp(text_emb)
        x = x + text_emb[:, :, None, None]
        
        x = F.relu(x)
        
        # Second conv
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x + residual)
        
        return x


class UNet(nn.Module):
    """UNet architecture for text-conditional diffusion model."""
    def __init__(self, 
                 image_channels=1,
                 base_channels=64,
                 time_embedding_dim=128,
                 text_embedding_dim=128,
                 num_classes=10):
        super().__init__()
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbeddings(time_embedding_dim)
        
        # Text embedding (for digit labels 0-9)
        self.text_embedding = TextEmbedding(num_classes, text_embedding_dim)
        
        # Initial projection
        self.initial_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder (downsampling)
        self.down1 = Block(base_channels, base_channels, time_embedding_dim, text_embedding_dim)
        self.down2 = Block(base_channels, base_channels * 2, time_embedding_dim, text_embedding_dim)
        self.down3 = Block(base_channels * 2, base_channels * 4, time_embedding_dim, text_embedding_dim)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = Block(base_channels * 4, base_channels * 4, time_embedding_dim, text_embedding_dim)
        
        # Decoder (upsampling) - using ConvTranspose2d for exact size matching
        # output_padding=1 for upconv1 to handle odd dimensions (7×7)
        self.upconv1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=2, stride=2, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        
        self.up1 = Block(base_channels * 8, base_channels * 2, time_embedding_dim, text_embedding_dim)
        self.up2 = Block(base_channels * 4, base_channels, time_embedding_dim, text_embedding_dim)
        self.up3 = Block(base_channels * 2, base_channels, time_embedding_dim, text_embedding_dim)
        
        # Final output
        self.final_conv = nn.Conv2d(base_channels, image_channels, kernel_size=1)
    
    def forward(self, x, timesteps, labels):
        """
        Args:
            x: Noisy image tensor [B, C, H, W]
            timesteps: Timestep tensor [B]
            labels: Digit labels [B] (0-9)
        """
        # Get embeddings
        time_emb = self.time_embedding(timesteps)
        text_emb = self.text_embedding(labels)
        
        # Initial conv
        x = self.initial_conv(x)
        
        # Encoder
        skip1 = self.down1(x, time_emb, text_emb)
        x = self.pool(skip1)
        
        skip2 = self.down2(x, time_emb, text_emb)
        x = self.pool(skip2)
        
        skip3 = self.down3(x, time_emb, text_emb)
        x = self.pool(skip3)
        
        # Bottleneck
        x = self.bottleneck(x, time_emb, text_emb)
        
        # Decoder with skip connections
        x = self.upconv1(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.up1(x, time_emb, text_emb)
        
        x = self.upconv2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up2(x, time_emb, text_emb)
        
        x = self.upconv3(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up3(x, time_emb, text_emb)
        
        # Final output (predict noise)
        x = self.final_conv(x)
        
        return x

