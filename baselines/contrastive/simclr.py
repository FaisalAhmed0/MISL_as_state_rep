import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torchvision

from config import cfg


# CNN Encoder
class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape, 32, 3, stride=2),
                                    nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                    nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                    nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                    nn.ReLU())

    def forward(self, obs):
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h
    

# SimClr
class SimCLR(nn.Module):
    def __init__(self, temp, cnn_encoder, in_channels, img_shape):
        super().__init__()
        self.temp = temp
        self.img_shape = img_shape
        self.cnn_backbone = cnn_encoder(in_channels) #torchvision.models.resnet18(zero_init_residual=True)
        # self.resnet_backbone.fc = nn.Identity()
        self.projector = nn.Sequential(nn.Linear(self.cnn_backbone.repr_dim,1024),
                                              nn.ReLU(),
                                              nn.Linear(1024,64))
        
        self.CEL = nn.CrossEntropyLoss(reduction="mean")
    def forward(self, x1, x2=None):
        if x2==None:
            return self.cnn_backbone(x1)

        else:
            z1 = self.projector(self.cnn_backbone(x1)) 
            z2 = self.projector(self.cnn_backbone(x2))

            loss = self.info_nce_loss(z1, z2, self.temp,self.CEL)

            return loss

    def info_nce_loss(self , z1, z2, temp , CEL):
        batch_size = z1.shape[0]
        device = z1.device
        z1 = nn.functional.normalize(z1, dim=1) #[b x d]
        z2 = nn.functional.normalize(z2, dim=1) #[b x d]
        z = torch.cat((z1, z2), axis=0) #[2*b x d]
        sim = torch.matmul(z, z.T) / temp  #[2*b x 2*b]
        # print('sim ',sim.shape)

        #We need to removed the similarities of samples to themselves
        off_diag_ids = ~torch.eye(2*batch_size, dtype=torch.bool, device=device)
        logits = sim[off_diag_ids].view(2*batch_size, -1)  #[2*b x 2*b-1]
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        labels = torch.cat([labels + batch_size - 1, labels])

        loss = CEL(logits, labels)
        return loss
