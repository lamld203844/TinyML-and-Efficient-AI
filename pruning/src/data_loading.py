"""Data loaders, datasets, and preprocessing scripts"""
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, RandomCrop, \
    RandomHorizontalFlip
from torch.utils.data import DataLoader

transforms = {
    'train': Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
    ]),
    'test': ToTensor(),
}

dataset = {}
for split in ['test', 'train']:
    dataset[split] = CIFAR10(
        root='data/cifar10',
        train=(split == 'train'),
        download=True,
        transform=transforms[split],
    )

dataflow = {}
for split in ['test', 'train']:
    dataflow[split] = DataLoader(
        dataset[split],
        batch_size=512,
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True,
    )
    