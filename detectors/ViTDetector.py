import torch.nn as nn
import torch
from torchvision import models

class ViTDetector(nn.Module):
    def __init__(self, pretrain=True, dropout=False):
        super(ViTDetector, self).__init__()
        self.dropout = dropout

        
        self.vit = models.vit_b_16(pretrained=pretrain)
        fc_dim = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(fc_dim, 9)

        train_layer = ['layer_11', 'head']
        for name, param in self.vit.named_parameters():
            if not any([layer in name for layer in train_layer]):
                param.requires_grad = False


    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)

        if self.dropout:
            c_in = self.drop(c_in)
        out = self.vit(c_in)
  
        return out
