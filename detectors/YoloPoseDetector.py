import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision.models import resnet34, ResNet34_Weights
from utils import *
from torchvision import transforms
import torch.nn.functional as F

class YoloPoseDetector(nn.Module):
    def __init__(self, n_dim=2048):
        super(YoloPoseDetector, self).__init__()

        yolo_model = YOLO("yolov8n-pose.pt").model
        self.yolo = yolo_model

        #self.backbone = yolo_model.model[:17]

        #self.hidden = None 
        #target_layer = self.yolo.model[21]  
        #target_layer.register_forward_hook(self.save_hidden_output)

        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        '''
        self.head = nn.Sequential(
            nn.Linear(1024, n_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(n_dim, 9)
        )
        '''

        '''
        train_layer = ['model.21', 'model.22']
        for name, param in self.yolo.named_parameters():
            if not any([layer in name for layer in train_layer]):
                param.requires_grad = False
            else:
                param.requires_grad = True
        '''
        #------- second net: resnet
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)

        res_layers = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*res_layers[:])

        #freeze_resnet(len(res_layers)-2, self.resnet)
        freeze_resnet(len(res_layers), self.resnet)


        
        self.head = nn.Sequential(
            #nn.Linear(1024 + resnet.fc.in_features, n_dim),
            nn.Linear(780, n_dim),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(n_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 9)
        )

        self.transform = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def save_hidden_output(self, module, input, output):
        if isinstance(output, tuple):
            self.hidden = output[0]
        else:
            self.hidden = output
 

    def adaptive_pool_flatten(self, tensor, output_size=(2, 2)):
        pooled = F.adaptive_avg_pool2d(tensor, output_size)  # [B, C, 2, 2]
        return pooled.flatten(start_dim=1)  # [B, C*2*2]

    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.shape
        x = x.view(batch_size * timesteps, C, H, W)
        #_ = self.yolo(x)
        #x1 = self.hidden
        #x1 = self.pool(x1)

        (x1a, x1b) = self.yolo(x)
        x_small, x_medium, x_large = x1a[0], x1a[1], x1a[2]
        x_small = self.adaptive_pool_flatten(x_small)
        x_medium = self.adaptive_pool_flatten(x_medium)
        x_large = self.adaptive_pool_flatten(x_large)
        features = torch.cat([x_small, x_medium, x_large], dim=1)

        #----
        x2 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x2 = self.transform(x2)
        x2 = self.resnet(x)
        x2 = x2.mean(3).mean(2)

        #out = torch.cat((x1, x2), dim=1)
        out = features
        out = self.head(out)

        return out

