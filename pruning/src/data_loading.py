"""Data loaders, datasets, and preprocessing scripts"""
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, RandomCrop, \
    RandomHorizontalFlip

transforms = {
    'train': Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
    ]),
    'test': ToTensor(),
}

dataflow = {}
for split in ['test', 'train']:
    dataflow[split] = CIFAR10(
        root='data/cifar10',
        train=(split == 'train'),
        download=True,
        transform=transforms[split],
    )

dataflow['train'], dataflow['test']