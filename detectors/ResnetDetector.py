import torch.nn as nn
import torch
from utils import freeze_resnet
from torchvision import transforms
import torch.nn.functional as F

class ResnetDetector(nn.Module):
    def __init__(self, modelName="resnet18", pretrain=True, dropout=False):
        super(ResnetDetector, self).__init__()
        self.modelName = modelName
        self.dropout = dropout
        
        match modelName:
            case "resnet18": 
                weights = "ResNet18_Weights"
            case "resnet34":
                weights = "ResNet34_Weights"
            case "resnet50":
                weights = "ResNet50_Weights"
            case "resnet152":
                weights = "ResNet152_Weights"
        weights = weights + ".DEFAULT"
        resnet = torch.hub.load('pytorch/vision:v0.10.0', modelName, weights=weights)
        fc_in_dim = resnet.fc.in_features
        res_layers = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*res_layers[:])

        print(f"res_layers  {len(res_layers)}")
        freeze_resnet(len(res_layers)-2, self.cnn)
        if self.dropout:
            self.drop = nn.Dropout(0.5)
        
        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 9)
        )

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        #print(f"batch = {batch_size}, time_step {timesteps}, C = {C} H = {H}, W = {W}")

        c_in = x.view(batch_size * timesteps, C, H, W)

        #---- remove later
        c_in  = F.interpolate(c_in, size=(224, 224), mode='bilinear', align_corners=False)
        transform = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        c_in = transform(c_in)
        #--- remove later end 
        
        # CNN forward
        c_out = self.cnn(c_in)
        #c_out = self.vit(c_in)
        #print(f"c_out {c_out.shape}")
        c_out = c_out.mean(3).mean(2) # cnn only
        if self.dropout:
            c_out = self.drop(c_out)
        #print(f"c_out {c_out.shape}")
        out = self.fc(c_out)
        out = out.view(batch_size*timesteps,9)       
        return out
    def getModelName(self):
        return self.modelName
