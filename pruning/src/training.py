"""Training loops, loss functions, optimizers"""
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(
    model: nn.Module,
    dataflow: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LambdaLR,
) -> None:
    """Train the model for one epoch.
    """

    model.train()
    
    for inputs, targets in tqdm(dataflow, desc='train', leave=False):
        # move from CPU to GPU
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Reset the gradients (from the last iteration)
        optimizer.zero_grad()
        
        # forward
        logits = model(inputs)
        loss = criterion(logits, targets)
        
        # backward propagation
        loss.backward()
        
        # Update optimizer and LR scheduler
        optimizer.step()
        scheduler.step()

        