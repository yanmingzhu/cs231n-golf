from golfdb.model import EventDetector
from golfdb.util import freeze_layers
from torchvision import models

from detectors.ResnetDetector import ResnetDetector
from detectors.ResnetLSTM import ResnetLSTM
from detectors.ViTDetector import ViTDetector
from detectors.ResnetTransformer import ResnetTransformer
from detectors.ResnetTransformer2 import ResnetTransformer2
from detectors.ViTTransformer import VITTransformer
from detectors.ViTLSTM import VITLSTM
from detectors.KeypointRCNN import KeypointRCNN
from detectors.ViTPoseDetector import ViTPoseDetector
from detectors.YoloPoseDetector import YoloPoseDetector
from detectors.ConvNext import ConvNext
from detectors.ConvNextTransformer import ConvNextTransformer
from detectors.ConvNextTransformer2 import ConvNextTransformer2


def getEventDetector():
    k = 10
    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    freeze_layers(k, model)

    return model

def getResnetLSTM():
    return ResnetLSTM(pretrain=True,
                      width_mult=1.,
                      lstm_layers=1,
                      lstm_hidden=256,
                      bidirectional=True,
                      dropout=False)

def get_model(model_name, dropout=0.1):
    match model_name:
        case "resnet18":
            return ResnetDetector(modelName="resnet18")
        case "resnet34":
            return ResnetDetector(modelName="resnet34")        
        case "resnet50":
            return ResnetDetector(modelName="resnet50")
        case "resnet152":
            return ResnetDetector(modelName="resnet152")
        case "swingnet":
            return getEventDetector()
        case "resnet-lstm":
            return getResnetLSTM()
        case "vit":
            return ViTDetector()
        case "resnet-transformer":
            return ResnetTransformer(dropout=dropout)
        case "resnet-transformer2":
            return ResnetTransformer2(dropout=dropout)
        case "vit-transformer":
            return VITTransformer()
        case "vit-lstm":
            return VITLSTM()
        case "rcnn":
            return KeypointRCNN()
        case "vitpose":
            return ViTPoseDetector()
        case "yolo":
            return YoloPoseDetector()
        case "convnext":
            return ConvNext()
        case "convnext-transformer":
            return ConvNextTransformer()
        case "convnext-transformer2":
            return ConvNextTransformer2()
        case _:
            print(f"Wrong model name {model_name}: check typo")
        
            