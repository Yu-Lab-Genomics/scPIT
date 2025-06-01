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
import os
import argparse

### script
from model import TransformerModel
from trainer import Trainer
from SingleCellDataset import SingleCellDataset
from create_dataloaders import create_dataloaders
from load_data import load_data
from set_seed import set_seed


### main
def main():
    # Argument parser to handle different modes
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                    default='../00.preprocessing/101_2981_4118.pth',
                    help='Path to the .pth dataset file, which is generated from a single-cell .h5ad file after preprocessing with the 00.perprocessing/Data_PreProcessing.ipynb script.')
    parser.add_argument('--gpu', type=int, default=5,
                        help='GPU index to use (default: 5)')
    args = parser.parse_args()


    # set seed # !! Due to the small dataset, variations in data distribution significantly impact model training. Therefore, we identified an optimal seed combination through large-scale screening ！！
    seed1 = 2 # training seed
    seed2 = 14 # data splitting seed

    # seed
    set_seed(2)

    # GPU
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # model config
    model_config = {
        'gene_count': 4118,  # number of genes
        'dim': 128,         # Transformer dim
    }

    # training config
    training_config = {
        'initial_lr': 0.000006,
        'batch_size': 12,       
        'num_epochs': 50,
        'patience':15
    }

    # load data set
    # Check if file exists
    data_path = args.data_path # .pth file from data preprocessing
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"File '{data_path}' not found.")
    # Check if it is a .pth file
    if not data_path.endswith('.pth'):
        raise ValueError(f"File '{data_path}' is not a .pth file.")
    print(f"Found .pth file: {data_path}")
    

    # training
    expr_tensor, expr_mask, disease_tensor, meta_tensor, celltype_tensor, target_tensor = load_data(data_path, device)
    dataset = SingleCellDataset(expr_tensor, expr_mask, disease_tensor, meta_tensor, celltype_tensor, target_tensor)
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=training_config["batch_size"],
        train_ratio=0.8,
        bins=6,
        seed=14
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

    # trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        regression_criterion=nn.MSELoss(),
        classification_criterion = nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        device=device,
        log_dir="./tensorboard",
        save_dir="./02.checkpoint",
        patience=training_config['patience'],
        mode='training'  # <-- pass mode to Trainer
    )

    # Run by mode
    cor, epoch, acc = trainer.train(training_config['num_epochs'])
    print(f"Training done. Best epoch: {epoch}, Pearson: {cor:.4f}, Acc: {acc:.4f}")

if __name__ == "__main__":
    main()