"""Main training script"""
import os
import sys
import random
import numpy as np
from tqdm import tqdm
# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join('.')))  
sys.path.append(os.path.abspath(os.path.join('..'))) 
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from src.data_loading import dataflow # 1. data
from src.models import VGG # 2. model
from src.training import train # 4. training loop
from src.evaluation import evaluate
# set random seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# logging
import logging
logging.basicConfig(
    level=logging.INFO,
    filename=f"./logs/train.log",
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
def log(message):
    logger.info(message)
    print(message)
# set gpu devive
torch.cuda.set_device(1)

NUM_EPOCHS = 20
SAVING_PATH = "./logs/saved_models/vgg.pth"

if __name__ == "__main__":

    # ------ 2. model ------
    model = VGG()
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
    steps_per_epoch = len(dataflow['train'])
    lamdba_lr = lambda step: np.interp(
        [step / steps_per_epoch],
        [0, NUM_EPOCHS * 0.3, NUM_EPOCHS],
        [0, 1, 0]
    )[0]
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lamdba_lr,
    )
    # ------ 4. training loop ------
    for epoch_num in tqdm(range(1, NUM_EPOCHS + 1)):
        train(model, dataflow["train"], criterion, optimizer, scheduler)
        metric = evaluate(model, dataflow["test"])
        log(f"\nepoch {epoch_num}: {metric:.2f}%")
        
    # ------ 5. saving ------
    torch.save(model.state_dict(), SAVING_PATH)
    log(f"Completed. Model saved to {SAVING_PATH}")
    
