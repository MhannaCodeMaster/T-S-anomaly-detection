import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18_MS3(nn.Module):

    def __init__(self, pretrained=False):
        super(ResNet18_MS3, self).__init__()     
        net = models.resnet18(pretrained=pretrained)
        # ignore the last block and fc
        self.model = torch.nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, x):
        res = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in ['4', '5', '6']:
                res.append(x)
        return res