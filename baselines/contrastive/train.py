import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import random
import os

from config import cfg
from utils import train
from utils import extract_data
from utils import seed_everything
from simclr import SimCLR, Encoder
import argparse

import wandb


seed_everything(5)


# Extract arguments from terminal
def cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--env', type=str, required=True)
    args = parser.parse_args()
    return args

def run(args):
    environment = args.env
    print(f"enviornment: {environment}")
    base_dir = "/mnt/qb/work/bethge/fmohamed65/MISL_as_state_rep/baselines/contrastive"
    base_data_dir = "/mnt/qb/work/bethge/fmohamed65"
    data_path = f"{base_data_dir}/exp_local/{args.data}/buffer/saved"
    
    
    # seting up wandb
    wandb.init(project="simclr", name=environment, entity="misi_as_state_rep", config=vars(cfg), dir=base_dir, settings=wandb.Settings(start_method='fork'))
    
    # These numbers are fixed for the dm_control
    in_channels = 9
    image_size = 84
    # Extract the data
    dmc_dataloader = extract_data(data_path)
    # create the model
    model = SimCLR(cfg.temp, Encoder, in_channels, image_size)
    print("############ SimCLr model ############")
    print(model)
    print("############ SimCLr model ############")
    # model optimizer
    save_path = f"{base_dir}/{environment}"
    os.makedirs(save_path, exist_ok=True)
    optimizer = opt.Adam(model.parameters(), cfg.lr, weight_decay=cfg.weight_decay)
    train(model, optimizer, cfg.epochs, dmc_dataloader, path=save_path)
    train(model, optimizer, cfg.epochs, dmc_dataloader, plot_freq=plot_freq, save_plots=True ,height=image_size, width=image_size, in_channels=in_channels, path=save_path, log=True)
    
    
if __name__ == "__main__":
    args = cmd_args()
    run(args)