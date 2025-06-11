import torch
import torch.nn as nn
from utils import freeze_resnet
from positional_emcoding import PositionalEncoding
from detectors.ConvNext import ConvNext

class ConvNextTransformer(nn.Module):
    def __init__(self, pretrain=True, n_head=8, n_dim=256, n_layers=6, dropout=0.1):
        super(ConvNextTransformer, self).__init__()
        self.n_head = n_head
        self.n_dim = n_dim

        #resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=pretrain)
        convnext_model = ConvNext()
        weight_file = 'models/ConvNext.pth.tar'
        save_dict = torch.load(weight_file)
        convnext_model.load_state_dict(save_dict['model_state_dict'])
        print(f"pretained weights for ConvNext is loaded")

        convnext_feature_dim = convnext_model.model.classifier[2].in_features
        convnext_model.model.classifier[2] = nn.Identity()
        for parameter in convnext_model.parameters():
            parameter.requires_grad = False

        self.convnext_model = convnext_model 
        self.proj = None
        if convnext_feature_dim != n_dim:
            self.proj = nn.Linear(convnext_feature_dim, n_dim)

        self.position_embed = PositionalEncoding(n_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_dim, nhead=self.n_head, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(nn.Linear(n_dim, 9))
 
    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()

        x = self.convnext_model(x)

        t_out = x.view(batch_size, timesteps, -1) 
        if self.proj != None: 
            t_out = self.proj(t_out)
        t_out = self.position_embed(t_out)
        t_out = self.transformer_encoder(t_out)
        out = self.head(t_out)

        out = out.view(batch_size*timesteps,9)

        return out

