'''
This file contains a convolutional VAE model
'''
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from config import cfg

class VAE(nn.Module):
    def __init__(self, in_channels=1, bottleneck=cfg.bottleneck, img_height=28, img_width=28):
        super(VAE, self).__init__()
        self.h = img_height
        self.w = img_width
        self.in_channels = in_channels

        # encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        ).to(cfg.device)

        # get the output shape
        self.img_shape, self.conv_out_shape = self._out_size()

        # linear layers for the mean and the log-variance
        self.mean = nn.Linear(self.conv_out_shape, bottleneck).to(cfg.device)
        self.logvar = nn.Linear(self.conv_out_shape, bottleneck).to(cfg.device)

        # projection
        self.linear = nn.Sequential(
            nn.Linear(bottleneck, self.conv_out_shape),
            nn.ReLU(),
        ).to(cfg.device)

        # decoder network
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=1,),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=1,),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 3, stride=1,),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 3, stride=2, output_padding=1),
            nn.Sigmoid()
        ).to(cfg.device)
    

    @torch.no_grad()
    def _out_size(self):
        x = torch.zeros(1, self.in_channels, self.h, self.w).to(cfg.device)
        x = self.encoder(x)
        return x.shape, torch.prod(torch.tensor(x.shape))

    def enc_forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar

    def dec_forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.img_shape[1], self.img_shape[2], self.img_shape[3])
        x = self.decoder(x)
        return x

    def reparametrization(self, mu, logvar):
        eps = torch.randn_like(mu).to(cfg.device)
        return mu + eps * logvar.exp().pow(0.5)

    def forward(self, x):
        # pass the input through the encoder
        mu, logvar = self.enc_forward(x)
        # sample from the latent space 
        z = self.reparametrization(mu, logvar)
        # pass z through the decoder
        x_hat = self.dec_forward(z)
        # print(x_hat.shape)
        return x_hat, mu, logvar

