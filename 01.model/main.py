import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import logging
import json
from tqdm import tqdm
from scipy import stats
from sklearn.model_selection import train_test_split
import sys

### script
from model import TransformerModel
from trainer import Trainer
from SingleCellDataset import SingleCellDataset
from create_dataloaders import create_dataloaders
from load_data import load_data
from set_seed import set_seed


### main
def main():
    # set seed # !! Due to the small dataset, variations in data distribution significantly impact model training. Therefore, we identified an optimal seed combination through large-scale screening ！！
    seed1 = 2 # training seed
    seed2 = 14 # data splitting seed

    # seed
    set_seed(int(seed1))

    # GPU
    device = torch.device(f'cuda:{6}' if torch.cuda.is_available() else 'cpu')

    # model config
    model_config = {
        'gene_count': 4118,  # number of genes
        'dim': 128,         # Transformer dim
    }

    # training config
    training_config = {
        'initial_lr': 0.000003,
        'batch_size': 24,       
        'num_epochs': 70,
        'patience':15
    }

    # load data set
    data_path = "101_2981_4118.pth" # .pth file from data preprocessing
    expr_tensor, expr_mask, disease_tensor, meta_tensor, celltype_tensor = load_data(data_path, device)
    dataset = SingleCellDataset(expr_tensor, expr_mask, disease_tensor, meta_tensor, celltype_tensor)
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=training_config["batch_size"],
        train_ratio=0.8,
        bins=6,
        seed=int(seed2)
    )

    # create model
    model = TransformerModel(
        gene_count=model_config['gene_count'],
        dim=model_config['dim']
    ).to(device)

    # Initial weight
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            if m.weight is not None:
                torch.nn.init.ones_(m.weight)
    model.apply(init_weights)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=training_config["initial_lr"])

    # tensorboard folder
    if not os.path.exists("./tensorboard"):
        os.makedirs("./tensorboard")

    # trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        criterion=nn.MSELoss(),
        optimizer=optimizer,
        device=device,
        log_dir="./tensorboard",
        save_dir="./02.checkpoint",
        patience=training_config['patience']
    )

    # training
    cor, epoch = trainer.train(training_config['num_epochs'])

    print(cor, epoch)

if __name__ == "__main__":
    main()