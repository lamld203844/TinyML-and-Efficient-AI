"""Main training script"""
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from src.data_loading import dataflow # 1. data
# model
from src.training import train # training loop
NUM_EPOCHS = 20
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

if __name__ == "__main__":

    # ------ 2. model ------
    
    # ------ 3. optimization ------
    # loss
    criterion = nn.CrossEntropyLoss() 
    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
    )   

    # scheduler
    #TODO
    
    # ------ 4. training loop ------
    for epoch_num in tqdm(range(1, NUM_EPOCHS + 1)):
        train(model, dataflow["train"], criterion, optimizer, scheduler)
        metric = evaluate(model, dataflow["test"])
        print(f"epoch {epoch_num}:", metric)