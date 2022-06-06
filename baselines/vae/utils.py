import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

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
import wandb

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



def extract_data(path):
    observatiobs = []
    files_names = os.listdir(path)
    length = 0
    for f in files_names:
        data = np.load(f"{path}/{f}")
        obs = torch.tensor(data['observation']).to(torch.float32).to(cfg.device)
        length += obs.shape[0]
        if length < 10000:
            observations.append(obs)
        else:
            break
    observations = torch.cat((observations), dim=0)
    # put the tensors in a dataset
    norm = transforms.Normalize([0.5]*9, [255]*9)
    y = torch.randn(obs.shape[0])
    dmc_dataset = CustomTensorDataset((obs, y), norm)
    dmc = DataLoader(dmc_dataset, batch_size=cfg.batch_size)
    return dmc_dataset, dmc

# load cifar10 for testing purposes
def load_cifar():
    cifar = CIFAR10(fmnist_root, train=True, download=True, transform=transforms.Compose([
                                                                                transforms.ToTensor(), 
                                                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) )
    cifar_test = CIFAR10(fmnist_root, train=False, download=True,  transform=transforms.Compose([
                                                                              transforms.ToTensor(), 
                                                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) )
    cifar_dataloader = DataLoader(cifar, batch_size=cfg.batch_size)
    cifar_test_dataloader = DataLoader(cifar_test, batch_size=cfg.batch_size)
    return cifar_dataloader, cifar_test_dataloader

# plot a batch of images as a grid.
def plot_grid(dataloader, batch=64, name="cifar", path="."):
    images, _ = next(iter(dataloader))
    images = images[:batch, :3, :, :]
    grid = make_grid(images, )
    plt.figure(figsize=(10, 10))
    filename = f"{path}/{name}.png"
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(filename)
    print(f"figure saved at: {filename}")
    
# VAE loss to optimize the ELBO
def vae_loss(output, target, mu, logvar):
    kl_divergence = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp())
#     loss = F.mse_loss(output, target, size_average=False) + kl_divergence
    loss = F.binary_cross_entropy(output, target, size_average=False) + kl_divergence
    return loss

# train function to fit the VAE
def train(model, optimizer, epochs, dataloader, plot_freq=10, save_plots=True, height=28, width=28, in_channels=1, path=".", log=False):
    model = model.to(cfg.device)
    losses = []
    test_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for img, _ in dataloader:
            img = img.to(cfg.device)
            output, mu, logvar = model(img)
            loss = vae_loss(output, img, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().item()

        target_output = img
        train_ouput = output.cpu().detach()
        epoch_loss /= len(dataloader)

        losses.append(epoch_loss)

    #     # evaluate on the test set
    #     with torch.no_grad():
    #         test_loss = 0
    #         for img, _ in test_dataloader:
    #             img = img.to(device)
    #             output, mu, logvar = model(img)
    #             loss = vae_loss(output, img, mu, logvar)
    #             test_loss += loss
    #         test_loss /= len(test_dataloader)
    #         test_losses.append(- test_loss)

        # plot some results every 10 epochs
        if (epoch+1) % plot_freq == 0 :
            targets = target_output[:min(64, cfg.batch_size), :3, :, :]
            output_reshaped = train_ouput.reshape(-1, in_channels, height, width)[:min(64, cfg.batch_size), :3, :, :]
            target_grid = make_grid(targets.cpu().detach(), nrow=8)
            output_grid = make_grid(output_reshaped.cpu().detach(), nrow=8)
            if log:
                plt.figure(figsize=(25, 20))
                plt.imshow(target_grid.permute(1, 2, 0))
                filename = f"{path}/target_grid.png"
                plt.savefig(filename)
                plt.figure(figsize=(25, 20))
                plt.imshow(output_grid.permute(1, 2, 0))
                filename = f"{path}/output_grid.png"
                plt.savefig(filename)
                if log:
                    wandb.log({"train/generated_images": wandb.Image(output_grid), "train/original_images": wandb.Image(target_grid)})
                    wandb.log({"epoch_loss": epoch_loss})

            # save the model
            model_filename = f"{path}/vae.ckpt"
            torch.save(model.state_dict(), model_filename)

        print(f"Epoch: {epoch+1}, train loss: {epoch_loss}")

    return losses







