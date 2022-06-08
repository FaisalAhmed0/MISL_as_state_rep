import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torchvision

from torch.utils.data import Dataset, DataLoader
from transforms import Transform
from config import cfg

import numpy as np
import random

import os
import wandb

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# TensorData with tranforms
class CustomTensorDataset(Dataset):
    # From https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
    
def extract_data(path, resize=True):
    observations = []
    files_names = os.listdir(path)
    length = 0
    print(f"file names: {files_names}")
    for f in files_names:
        if ".npz" in f:
            data = np.load(f"{path}/{f}")
            obs = torch.tensor(data['observation']).to(torch.float32).to(cfg. device)
            length += obs.shape[0]
            if length < 10000:
                observations.append(obs)
            else:
                break
    observations = torch.cat((observations), dim=0)
    print(observations.shape)
    # put the tensors in a dataset
    if resize:
        transform = Transform()
    else:
        transform = transforms.Normalize([0.5]*9, [255]*9)
    y = torch.randn(observations.shape[0])
    dmc_dataset = CustomTensorDataset((observations, y), transform)
    dmc = DataLoader(dmc_dataset, cfg. batch_size, shuffle=True)
    return dmc


def train(model, optimizer, epochs, dataloader,  path=".", log=True):
    model = model.to(cfg. device)
    losses = []
    test_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for ((x1, x2), _) in dataloader:
            x1 = x1.to(cfg. device)
            x2 = x2.to(cfg. device)
            loss = model(x1, x2)
            # print(loss.shape)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().item()

        # target_output = img
        # train_ouput = output.cpu().detach()
        epoch_loss /= len(dataloader)

        losses.append(epoch_loss)
        
        if log:
            wandb.log({"epoch_loss": epoch_loss})

        # save the model
        model_filename = f"{path}/cl.ckpt"
        torch.save(model.cnn_backbone.state_dict(), model_filename)

        print(f"Epoch: {epoch+1}, train loss: {epoch_loss}")

    return losses