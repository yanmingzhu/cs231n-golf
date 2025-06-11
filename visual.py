import matplotlib.pyplot as plt
from datasets import get_training_data_loader
import torch

if __name__ == '__main__':
    dl = get_training_data_loader(batch_size=1, seq_length=1)

    for i, item in enumerate(dl):
        image = item["images"][0][0]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        unnormalized = image * std + mean

        image = unnormalized.permute(1, 2, 0)
        img_np = unnormalized.permute(1, 2, 0).numpy()

        plt.subplot(1, 10, i+1)
        plt.imshow(image)
        plt.axis('off')

        if i >= 9:
            break
    plt.show()