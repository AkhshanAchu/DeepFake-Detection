import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .mvit import MViT
from .cmf import CMFBlock


class MViT_combined_cmf(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, n_blocks=4):
        super(MViT_combined_cmf, self).__init__()
        self.image_size = 224
        self.patch_size = 16
        self.in_channels = 3
        self.mvit_block = MViT(self.image_size, self.patch_size, self.in_channels)
        self.cmf_block = CMFBlock(input_channels=5)
        
    def forward(self, x):
        out = self.mvit_block(x)
        out = self.cmf_block(out, x)
        return x


class CMVit_repeat(nn.Module):
    def __init__(self, num_classes=2, n_blocks=6, embedding_dim=256):
        super(CMVit_repeat, self).__init__()
        
        self.image_size = 224
        self.patch_size = 16
        self.in_channels = 3
        self.n_blocks = n_blocks
        self.embedding_dim = embedding_dim
        self.conv = nn.AdaptiveAvgPool2d((100, 100))

        self.blocks_mvit = nn.ModuleList([MViT_combined_cmf(self.image_size, self.patch_size, self.in_channels) for _ in range(n_blocks)])

    def forward(self, x):
        for block in self.blocks_mvit:
            x = block(x)

        x = self.conv(x)
        return x


class classifier_block(nn.Module):
    def __init__(self, num_classes=2, n_blocks=6, device="cuda"):
        super(classifier_block, self).__init__()
        self.model_backbone = CMVit_repeat(num_classes=num_classes, n_blocks=n_blocks)
        self.flatten = nn.Flatten()
        self.normal = nn.GELU()
        base_model = getattr(models, "convnext_base")(pretrained=True)
        self.feature_model = nn.Sequential(*list(base_model.children())[:-1]).to(device)
        self.feature_model.eval()
        
        self.fc1 = nn.Linear(30000 + 1024, 1000)
        self.fc2 = nn.Linear(1000, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(p=0.5)

    def feature_extractor(self, images, model):
        with torch.no_grad():
            features = model(images)
        features = features.flatten(start_dim=1)
        return features
        
    def forward(self, x):
        features = self.feature_model(x).squeeze([2, 3])
        x = self.model_backbone(x)
        x = self.flatten(x)
        x = torch.concat([x, features], dim=1)
        
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        x = F.log_softmax(x, dim=-1)
        return x
