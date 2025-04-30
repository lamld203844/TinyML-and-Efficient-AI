"""Metrics, evaluation scripts, visualization tools"""
from src.data_loading import dataflow

# --------- visualization dataset ---------
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
    
if __name__ == "__main__":
    # visualize_cifar(dataflow['train'])
    # visualize_cifar(dataflow['test'])
    pass