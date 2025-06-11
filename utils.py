import argparse
import torch
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str, 
                        choices=['swingnet', 
                                 'resnet18',
                                 'resnet34',
                                 'resnet50',
                                 'resnet152', 
                                 'resnet-lstm',
                                 'vit-lstm',
                                 'vit',
                                 'resnet-transformer',
                                 'resnet-transformer2',
                                 'vit-transformer',
                                 'rcnn',
                                 'vitpose',
                                 'yolo',
                                 'convnext',
                                 'convnext-transformer',
                                 'convnext-transformer2'
                                 ])

    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seqlen", type=int, default=64)
    parser.add_argument("--batch", type=int, default=22)
    parser.add_argument("--drop", type=float, default=0.1)
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--data_remote")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--start_weight", type=str)
    parser.add_argument("--val", action='store_true')
    print("---valll2--")
    args = parser.parse_args()
    return args

def check_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device

def get_model_name(model):
    if hasattr(model, "getModelName"):
        return model.getModelName()
    else:
        return type(model).__name__

def freeze_resnet(n_layers, resnet):
    for i, layer in enumerate(resnet):
        if i < n_layers:
            for param in layer.parameters():
                param.requires_grad = False


def show_image(x, limit=100):
    if len(x.shape) > 4:
        B, T, C, H, W= x.shape
        images = x.reshape(B * T, C, H, W)
    else:
        images = x

    n_images = images.shape[0]
    print(f"n_images={n_images}")

    print(f"images {images.shape} on device {images.device}")
    for i in range(min(limit, n_images)):
        '''
        image = item["images"][0][0]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        unnormalized = image * std + mean
        '''
        #image = images[i].permute(1, 2, 0)
        image =  images[i].permute(1, 2, 0).cpu().numpy()
        print(f"image {image.shape} on device {image.device}")
        plt.subplot(1, n_images, i+1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()       


def get_class_weights():
    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    class_label_count = torch.tensor([50795, 61820, 64868, 67011, 65498, 63317, 61239, 40940, 11812512]).to(check_device()) # class samples obtained from running a 3000 iteration 
    #class_weights = class_label_count.sum() / class_label_count / 9

    #class_weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).to(device)

    # normalized  Logarithmic Inverse Frequency 
    mu = 0.3
    class_weights = torch.log(mu * class_label_count.max() / class_label_count + 1)
    class_weights = class_weights / class_weights.max()
    print(f"class_weights = {class_weights}")
    return class_weights
