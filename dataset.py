from enum import Enum
from typing import Tuple

import PIL
import torch.utils.data
from PIL import Image
from torch import Tensor
from torchvision import datasets
from torchvision.transforms import transforms

class Dataset(torch.utils.data.Dataset):

    class Mode(Enum):
        TRAIN = 'train'
        TEST = 'test'

    def __init__(self, path_to_data_dir: str, mode: Mode):
        super().__init__()
        is_train = mode == Dataset.Mode.TRAIN
        data_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dataset = datasets.ImageFolder(root='data/processed',transform=data_transform)

    def preprocess(image: PIL.Image.Image) -> Tensor:
        # TODO: CODE BEGIN
        #raise NotImplementedError
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)
        return image
        #self._cifar10 = datasets.CIFAR10(path_to_data_dir, train=is_train, download=True)
        #dataset = datasets.ImageFolder(root='data/processed',transform=data_transform)
