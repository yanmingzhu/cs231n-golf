
from transformers.models.vitpose import VitPoseImageProcessor, VitPoseForPoseEstimation
import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)
from utils import check_device, show_image

class ViTPoseDetector(nn.Module):
    def __init__(self, n_dim=768):
        super(ViTPoseDetector, self).__init__()
        device = check_device()
        self.person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        self.person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)
        for param in self.person_model.parameters():
            param.requires_grad = False

        self.vitpose = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
        self.image_processor = VitPoseImageProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        for param in self.vitpose.parameters():
            param.requires_grad = False

        train_layer = ['layer.11']
        for name, param in self.vitpose.named_parameters():
            if not any([layer in name for layer in train_layer]):
                param.requires_grad = False

        self.head = nn.Linear(n_dim, 9)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(x.device)[None, :, None, None]

        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        #show_image(x)
        #x = x * std + mean
        
        # CNN forward
        x = x.view(batch_size * timesteps, C, H, W)

        images_processed = self.person_image_processor(images=x, 
                                                       #size={"height": 224, "width": 224}, 
                                                       return_tensors="pt").to(x.device)
        with torch.no_grad():
            results = self.person_model(**images_processed)
        #print(f"results1 = {results}")
        image_size = torch.tensor([224, 224]).to(x.device)
        image_size = image_size.unsqueeze(0)
        image_size = image_size.repeat(batch_size*timesteps, 1)
        results = self.person_image_processor.post_process_object_detection(
            results, target_sizes=image_size, threshold=0.001)
        
        '''
        for i, result in enumerate(results):
            for box, label, score in zip(results[0]['boxes'], results[0]['labels'], results[0]['scores']):
                if label.item() == 0:
                    print(f"image:{i} Label: {label.item()}, Score: {score.item():.2f}")
        '''
        # Human label refers 0 index in COCO dataset
        person_boxes = []
        for result in results:
            scores = result['scores']
            scores[result["labels"] != 0] = float('-inf')
            box = result["boxes"][scores.argmax()]
            box = box.unsqueeze(dim=0)
            box[:, 2] = box[:, 2] - box[:, 0]
            box[:, 3] = box[:, 3] - box[:, 1]
            person_boxes.append(box.cpu().numpy())

        image_processed = self.image_processor(images=x, 
                                               #size={"height": 224, "width": 224}, 
                                               boxes=person_boxes, return_tensors="pt").to(x.device)

        output = self.vitpose(**image_processed, output_hidden_states=True)
        hidden_states = output.hidden_states
        features = hidden_states[-1]

        features = torch.mean(features, dim=1)

        out = self.head(features)

        return out