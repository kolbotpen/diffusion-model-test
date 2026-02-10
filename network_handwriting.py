import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
import numpy as np
import math


class SinusoidalPositionEmbeddings(nn.Module):
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


def get_sinusoidal_position_encoding(seq_len, dim, device, batch_size=1):
    """Generate sinusoidal position encodings"""
    position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    
    pe = torch.zeros(seq_len, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    if dim % 2 == 1:
        pe[:, 1::2] = torch.cos(position * div_term[:, :-1])
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    
    # Return with batch dimension - use repeat for proper gradient flow
    return pe.unsqueeze(0).repeat(batch_size, 1, 1)


class TextEncoder(nn.Module):
    """Encodes text as a sequence with positional information preserved"""
    def __init__(self, vocab_size=128, embed_dim=256, max_seq_len=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
    def forward(self, char_indices):
        # char_indices: (batch, seq_len)
        batch_size, seq_len = char_indices.shape
        
        # Character embeddings
        char_emb = self.embedding(char_indices)  # (batch, seq_len, embed_dim)
        
        # Add sinusoidal positional encodings
        pos_enc = get_sinusoidal_position_encoding(seq_len, self.embed_dim, char_indices.device, batch_size)
        text_seq = char_emb + pos_enc  # (batch, seq_len, embed_dim)
        
        return text_seq  # Return as sequence, not averaged


class ConditionalAffineTransform(nn.Module):
    """Affine transformation conditioned on time/noise level"""
    def __init__(self, channels, condition_dim):
        super().__init__()
        self.scale_shift = nn.Linear(condition_dim, channels * 2)
        
    def forward(self, x, condition):
        # x: (batch, channels, height, width)
        # condition: (batch, condition_dim)
        scale_shift = self.scale_shift(condition)
        scale, shift = scale_shift.chunk(2, dim=1)
        
        # Apply affine transformation along channel axis
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x


class ConvBlock(nn.Module):
    """Convolutional block with 3 conv layers and conditional affine transformations"""
    def __init__(self, channels_in, channels_out, time_embedding_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, channels_out)
        self.norm2 = nn.GroupNorm(8, channels_out)
        self.norm3 = nn.GroupNorm(8, channels_out)
        
        # Conditional affine transformations
        self.affine1 = ConditionalAffineTransform(channels_out, time_embedding_dim)
        self.affine2 = ConditionalAffineTransform(channels_out, time_embedding_dim)
        self.affine3 = ConditionalAffineTransform(channels_out, time_embedding_dim)
        
        self.residual_conv = nn.Conv2d(channels_in, channels_out, kernel_size=1) if channels_in != channels_out else nn.Identity()
    
    def forward(self, x, time_emb):
        residual = self.residual_conv(x)
        
        # First conv
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.affine1(x, time_emb)
        x = F.relu(x)
        
        # Second conv
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.affine2(x, time_emb)
        x = F.relu(x)
        
        # Third conv
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.affine3(x, time_emb)
        
        return F.relu(x + residual)


class AttentionBlock(nn.Module):
    """Lightweight attention block with only cross-attention to text"""
    def __init__(self, channels, text_dim, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # Cross-attention to text only (no self-attention for speed)
        self.cross_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(channels)
        
        # Simplified feed-forward network (2x instead of 4x)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels)
        )
        self.ffn_norm = nn.LayerNorm(channels)
        
        # Project text to match channel dimension
        self.text_proj = nn.Linear(text_dim, channels)
        
    def forward(self, x, text_seq, time_emb):
        # x: (batch, channels, H, W)
        # text_seq: (batch, seq_len, text_dim)
        
        batch, channels, h, w = x.shape
        device = x.device
        
        # Reshape to sequence
        x_seq = x.view(batch, channels, h * w).permute(0, 2, 1)  # (batch, H*W, channels)
        
        # Add positional encodings
        spatial_seq_len = h * w
        text_seq_len = text_seq.shape[1]
        
        spatial_pos = get_sinusoidal_position_encoding(spatial_seq_len, channels, device, batch)
        
        # Cross-attention with text (only cross-attention, no self-attention)
        text_proj = self.text_proj(text_seq)  # (batch, seq_len, channels)
        text_pos = get_sinusoidal_position_encoding(text_seq_len, channels, device, batch)
        
        q = x_seq + spatial_pos
        k = text_proj + text_pos
        v = text_proj
        
        attn_out, _ = self.cross_attn(q, k, v)
        x_seq = self.cross_attn_norm(x_seq + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x_seq)
        x_seq = self.ffn_norm(x_seq + ffn_out)
        
        # Reshape back to spatial
        x = x_seq.permute(0, 2, 1).view(batch, channels, h, w)
        
        return x


class UNetHandwriting(nn.Module):
    """UNet architecture for handwriting generation (64x256 images) with attention"""
    def __init__(self, 
                image_channels=1,
                base_channels=64,
                time_embedding_dim=256,
                text_embedding_dim=256,
                vocab_size=128,
                max_seq_len=32,
                num_heads=8):
        super().__init__()
        
        self.time_embedding = SinusoidalPositionEmbeddings(time_embedding_dim)
        self.text_encoder = TextEncoder(vocab_size, text_embedding_dim, max_seq_len)
        
        # Time embedding MLP (2 fully connected layers as per paper)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.GELU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim)
        )
        
        # Initial projection
        self.initial_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder (downsampling) - using ConvBlocks + AttentionBlocks
        # Note: Attention only at lowest resolutions to save computation
        self.down_conv1 = ConvBlock(base_channels, base_channels, time_embedding_dim)
        # No attention at 64x256
        
        self.down_conv2 = ConvBlock(base_channels, base_channels * 2, time_embedding_dim)
        # No attention at 32x128
        
        self.down_conv3 = ConvBlock(base_channels * 2, base_channels * 4, time_embedding_dim)
        # No attention at 16x64 (removed for speed)
        
        self.down_conv4 = ConvBlock(base_channels * 4, base_channels * 8, time_embedding_dim)
        self.down_attn4 = AttentionBlock(base_channels * 8, text_embedding_dim, num_heads)  # 8x32 only
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck_conv = ConvBlock(base_channels * 8, base_channels * 8, time_embedding_dim)
        self.bottleneck_attn = AttentionBlock(base_channels * 8, text_embedding_dim, num_heads)
        
        # Decoder (upsampling)
        self.upconv1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(base_channels * 16, base_channels * 4, time_embedding_dim)
        self.up_attn1 = AttentionBlock(base_channels * 4, text_embedding_dim, num_heads)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(base_channels * 8, base_channels * 2, time_embedding_dim)
        # No attention at 16x64 (removed for speed)
        
        self.upconv3 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=2, stride=2)
        self.up_conv3 = ConvBlock(base_channels * 4, base_channels, time_embedding_dim)
        # No attention at 32x128 - too memory intensive
        
        self.upconv4 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.up_conv4 = ConvBlock(base_channels * 2, base_channels, time_embedding_dim)
        # No attention at 64x256 - too memory intensive
        
        # Final output
        self.final_conv = nn.Conv2d(base_channels, image_channels, kernel_size=1)
    
    def forward(self, x, timesteps, text_indices):
        # Get embeddings
        time_emb = self.time_embedding(timesteps)
        time_emb = self.time_mlp(time_emb)  # Process through MLP
        
        text_seq = self.text_encoder(text_indices)  # (batch, seq_len, text_dim)
        
        # Initial conv
        x = self.initial_conv(x)
        
        # Encoder with skip connections
        skip1 = self.down_conv1(x, time_emb)  # 64x256 - no attention
        x = self.pool(skip1)
        
        skip2 = self.down_conv2(x, time_emb)  # 32x128 - no attention
        x = self.pool(skip2)
        
        skip3 = self.down_conv3(x, time_emb)  # 16x64 - no attention
        x = self.pool(skip3)
        
        skip4 = self.down_conv4(x, time_emb)  # 8x32 - with attention
        skip4 = self.down_attn4(skip4, text_seq, time_emb)
        x = self.pool(skip4)
        
        # Bottleneck
        x = self.bottleneck_conv(x, time_emb)
        x = self.bottleneck_attn(x, text_seq, time_emb)
        
        # Decoder with skip connections
        x = self.upconv1(x)
        x = torch.cat([x, skip4], dim=1)
        x = self.up_conv1(x, time_emb)
        x = self.up_attn1(x, text_seq, time_emb)
        
        x = self.upconv2(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.up_conv2(x, time_emb)  # 16x64 - no attention
        
        x = self.upconv3(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up_conv3(x, time_emb)  # 32x128 - no attention
        
        x = self.upconv4(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up_conv4(x, time_emb)  # 64x256 - no attention
        
        # Final output
        x = self.final_conv(x)
        
        return x
