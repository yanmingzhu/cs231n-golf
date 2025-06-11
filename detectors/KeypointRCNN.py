import torchvision
import torch.nn as nn
from torchvision.models.detection import keypointrcnn_resnet50_fpn

class KeypointRCNN(nn.Module):
    def __init__(self):
        super(KeypointRCNN, self).__init__()
    
        keypoint_rcnn = keypointrcnn_resnet50_fpn(pretrained=True)
        self.backbone = keypoint_rcnn.backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.flatten = nn.Flatten()
        self.head = nn.Linear(256, 9)

        train_layer = ['fpn']#, 'layer4']
        for name, param in self.backbone.named_parameters():
            if not any([layer in name for layer in train_layer]):
                param.requires_grad = False
    
    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)
        x = features['0']
        x = self.pool(x)
        x = self.flatten(x)
        out = self.head(x)

        #out = out.reshape(B, T, -1)
        return out
