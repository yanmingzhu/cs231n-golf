import torch.nn as nn
import torch
from timm import create_model
from torchvision.models import convnext_tiny, convnext_small, ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights

class ConvNext(nn.Module):
    def __init__(self, n_dim=256):
        super(ConvNext, self).__init__()

        #model = create_model('convnext_small', pretrained=True)
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        print(f"convnext {model}")
        
        '''
        model.head.fc = nn.Sequential(
            nn.Linear(model.head.in_features, n_dim),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(n_dim, 9)
        ) 
        '''
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, 9)

        self.model = model

        train_layer = ['classifier', '7.2']       
        for name, param in self.model.named_parameters():
            if not any([layer in name for layer in train_layer]):
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        #print(f"batch = {batch_size}, time_step {timesteps}, C = {C} H = {H}, W = {W}")

        x = x.view(batch_size*timesteps, C, H, W)
        out = self.model(x)

        return out
