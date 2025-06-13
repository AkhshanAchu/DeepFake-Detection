import torch
import torch.nn as nn
import torch.nn.functional as F


class MViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hid_dim, dropout=0.1):
        super(MViTBlock, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hid_dim, embed_dim),
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.multihead_attention(x, x, x)
        x = self.layer_norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)

        return x


class MViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim=256, num_blocks=12, num_heads=8, ff_hid_dim=512, dropout=0.1):
        super(MViT, self).__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        self.blocks = nn.ModuleList([MViTBlock(embed_dim, num_heads, ff_hid_dim, dropout) for _ in range(num_blocks)])

        self.conv_transpose = nn.ConvTranspose2d(embed_dim, 3, kernel_size=16, stride=16)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        x = x + self.positional_embedding

        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2).reshape(-1, self.embed_dim, 14, 14)

        x = self.conv_transpose(x)

        return x
