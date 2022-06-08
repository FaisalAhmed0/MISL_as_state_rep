import torch
from argparse import Namespace
# configiration variables
cfg = Namespace(
# batch size
batch_size = 256,
# learning rate
lr = 1e-4,
# epochs
epochs = 500,
# weight_decay
weight_decay = 0,
# temperature
temp = 0.5,
# device
device = "cuda" if torch.cuda.is_available() else "cpu"
)