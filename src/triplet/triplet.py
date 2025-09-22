import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TripletEmbedder(nn.Module):
    def __init__(self, embed_dim=128, pretrained=True, freeze_backbone=False):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = m.fc.in_features 
        m.fc = nn.Identity()
        self.backbone = m
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = nn.Linear(feat_dim, embed_dim)

    def forward(self, x):
        z = self.backbone(x)     # [B,512]
        z = self.head(z)         # [B,embed_dim]
        return F.normalize(z, p=2, dim=1)  # unit-norm for cosine
