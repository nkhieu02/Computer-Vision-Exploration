import PIL.Image
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.v2 import functional as F

def show(sample, transform):
    image, target = sample
    print(target)
    if isinstance(image, PIL.Image.Image):
        image = F.to_image(image)
    image = F.to_dtype(image, torch.float32, scale=True)
    if not transform == None:
        image = transform(image)
    fig, ax = plt.subplots()
    ax.imshow(image.permute(1, 2, 0))
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()