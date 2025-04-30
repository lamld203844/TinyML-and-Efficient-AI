"""Metrics, evaluation scripts, visualization tools"""
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Add parent directory to path
import os, sys
sys.path.append(os.path.abspath(os.path.join('.')))  
sys.path.append(os.path.abspath(os.path.join('..'))) 
from src.data_loading import dataflow
from src.models import VGG
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------- visualizing dataset ---------
# draw grid 4x10, image x class
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

def visualize_cifar(dataflow: CIFAR10):
    """Visualize the dataset with 4 rows and 10 columns."""
    # get 40 samples from the training set, 10 classes = 10 list, each list has 
    # 4 images
    samples = [[] for i in range(10)]
    for img, label in dataflow:
        if len(samples[label]) < 4:
            samples[label].append(img)

    plt.figure(figsize=(20, 9))
    for idx in range(40):
        # get corresponding image, label (class ~ column)
        row = idx // 10
        col = idx % 10
        img = samples[col][row]
        label = dataflow.classes[col] 
        
        plt.subplot(4, 10, idx + 1)
        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        plt.title(label)
        plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
# --------- evaluation ---------
@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataflow: DataLoader,
) -> float:
    model.eval()
    model.to(device)
    samples = 0
    correct = 0
    for inputs, labels in tqdm(dataflow, desc='eval', leave=False):
        # move data to GPU (if possible)
        inputs, labels = inputs.to(device), labels.to(device)

        # forward
        logits = model(inputs) # (b, cls)
        preds = logits.argmax(dim=1)
        
        # eval metric computing: accuracy
        samples += inputs.shape[0]
        correct += (preds == labels).sum().item()
    
    return correct / samples * 100  
    
if __name__ == "__main__":
    # visualize_cifar(dataflow['train'])
    
    # test evaluate func
    model = VGG()
    acc = evaluate(model, dataflow['test'])
    print(f"Accuracy: {acc:.2f}%")
    assert type(acc) == float
    
    # visualize_cifar(dataflow['test'])
    