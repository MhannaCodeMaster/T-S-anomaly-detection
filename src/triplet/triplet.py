import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Sampler
import random

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

class StratifiedTwoClassBatchSampler(Sampler):
    """Yields indices for batches containing half OK and half NOT_OK."""
    def __init__(self, len_ok, len_ng, batch_size, drop_last=False):
        assert batch_size % 2 == 0, "Use even batch_size"
        self.len_ok, self.len_ng = len_ok, len_ng
        self.bs = batch_size
        self.drop_last = drop_last
        self.ok_idx = list(range(0, len_ok))
        self.ng_idx = list(range(len_ok, len_ok+len_ng))

    def __iter__(self):
        ok = self.ok_idx[:]
        ng = self.ng_idx[:]
        random.shuffle(ok); random.shuffle(ng)
        i = j = 0
        while i + self.bs//2 <= len(ok) and j + self.bs//2 <= len(ng):
            batch = ok[i:i+self.bs//2] + ng[j:j+self.bs//2]
            random.shuffle(batch)
            yield batch
            i += self.bs//2; j += self.bs//2
        if not self.drop_last:
            # handle remainders (optional: top-up with random)
            pass

    def __len__(self):
        return min(self.len_ok, self.len_ng) * 2 // self.bs
