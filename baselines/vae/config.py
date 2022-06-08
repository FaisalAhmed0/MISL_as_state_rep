import torch
from argparse import Namespace
# configiration variables
cfg = Namespace(
# learning rate
lr = 1e-3,
# weight decay
weight_decay = 0.0,
# batch size
batch_size = 256,
# size of the latent space vector
bottleneck = 1024,
# weight for the KL loss
kl_wight = 1,
# number of epochs
epochs = 1000,
# device
device = 'cuda'
)