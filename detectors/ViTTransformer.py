import torch
import torch.nn as nn
import torchvision#
import torchvision.transforms as transforms
from torchvision import models

class VITTransformer(nn.Module):
    def __init__(self, pretrain=True, n_head=8, n_dim=768, n_layers=1, dropout=True):
        super(VITTransformer, self).__init__()
        self.n_head = n_head
        self.n_dim = n_dim
        self.dropout = dropout

        net = models.vit_b_16(pretrained=pretrain)
        net.heads = nn.Identity()

        self.vit = net
        train_layer = []
        for name, param in self.vit.named_parameters():
            if not any([layer in name for layer in train_layer]):
                param.requires_grad = False
        
        if self.dropout:
            self.drop = nn.Dropout(0.5)
    
        self.transformer_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=n_dim, nhead=self.n_head) for _ in range(n_layers)
        ])

        self.linear = nn.Sequential(
            nn.Linear(n_dim, 9)
        )

 
    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        #print(f"x={x.shape} c_in={c_in.shape}")

        c_out = self.vit(c_in)
        #print(f"c_out={c_out.shape}")
        if self.dropout:
            c_out = self.drop(c_out)

        # transformer forward
        t_out = c_out.view(batch_size, timesteps, -1) #+ pos_embedding
        for layer in self.transformer_encoder:
            t_out = layer(src=t_out)
        out = self.linear(t_out)

        #print(f"c_out {c_out.shape} t_out {t_out.shape}")

        out = out.view(batch_size*timesteps,9)

        return out

