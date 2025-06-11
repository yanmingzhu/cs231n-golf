import torch
import torch.nn as nn
from utils import freeze_resnet
from positional_emcoding import PositionalEncoding
from torchvision.models import resnet34, ResNet34_Weights

class ResnetTransformer2(nn.Module):
    def __init__(self, pretrain=True, n_head=8, n_dim=256, n_layers=2, dropout=0.1):
        super(ResnetTransformer2, self).__init__()
        self.n_head = n_head
        self.n_dim = n_dim

        #resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=pretrain)
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)

        res_layers = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*res_layers[:])

        train_layer = ['7.2', '7.1']       
        for name, param in self.cnn.named_parameters():
            if not any([layer in name for layer in train_layer]):
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        #freeze_resnet(len(res_layers)-2, self.cnn)
        #freeze_resnet(7, self.cnn)
        
        self.proj = None
        if resnet.fc.in_features != n_dim:
            self.proj = nn.Linear(resnet.fc.in_features, n_dim)

        self.position_embed = PositionalEncoding(n_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_dim, nhead=self.n_head, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(nn.Linear(n_dim, 9))
 
    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)

        # transformer forward
        t_out = c_out.view(batch_size, timesteps, -1) 
        if self.proj != None: 
            t_out = self.proj(t_out)
        t_out = self.position_embed(t_out)
        t_out = self.transformer_encoder(t_out)
        out = self.head(t_out)

        out = out.view(batch_size*timesteps,9)

        return out

