import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torchvision import transforms
from agent.cic import RMS



# CNN encoder (shared between RL and representation learning)
class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h
    
    
class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
          transforms.RandomResizedCrop(84),
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.RandomApply(
              [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.2, hue=0.1)],
              p=0.8
          ),
          transforms.RandomGrayscale(p=0.2),
          transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
#           Solarization(p=0.0),
#           transforms.ToTensor(),
#           transforms.Normalize(
#                   mean=(0.5)*9, std=(0.5)*9)
        ]
        )
        self.transform_prime = transforms.Compose([
          transforms.RandomResizedCrop(84),
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.RandomApply(
              [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.2, hue=0.1)],
              p=0.8
          ),
          transforms.RandomGrayscale(p=0.2),
          transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
#           Solarization(p=0.2),
#           transforms.ToTensor(),
#           transforms.Normalize(
#                   mean=(0.5)*9, std=(0.5)*9)
        ]
        )

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2
    
    
# SimCLR projection
class SimCLR(nn.Module):
    def __init__(self, temp, cnn_encoder, latent_size, hidden_dim):
        super().__init__()
        self.temp = temp
        self.cnn_backbone = cnn_encoder
        # self.resnet_backbone.fc = nn.Identity()
        self.projector = nn.Sequential(nn.Linear(self.cnn_backbone.repr_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                       nn.Linear(hidden_dim,latent_size))
        self.transform = Transform()
        
        self.CEL = nn.CrossEntropyLoss(reduction="mean")
        # running mean and std of the contrastive reward
        self.rms = RMS()
        
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
    
    def info_nce_reward(self, x1, x2):
        batch_size = x1.shape[0]
        device = x1.device
        
        z1 = self.projector(self.cnn_backbone(x1)) 
        z2 = self.projector(self.cnn_backbone(x2))
        
        z1 = nn.functional.normalize(z1, dim=1) #[b x d]
        z2 = nn.functional.normalize(z2, dim=1) #[b x d]
        # calulate the similarity
        sim = - (torch.sum(z1 * z2, dim=1)) # shape: (B,)
#         print(f"sim: {sim[:64]}")
        reward = sim + 1.2
        runnung_mean, running_std = self.rms(reward)
#         print(f"reward: {reward}")
#         print(f"running_std: {running_std}")
        return reward
    
    
# In case of using CIC the reward will come from the entropy term
class CIC(nn.Module):
    def __int__(self, cnn_encoder, obs_dim, skill_dim, latent_size, temp, hidden_dim):
        # obs dim in the image-based case it is the size of the cnn_encoder output
        self.cnn_encoder = cnn_encoder
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim
        self.latent_size = latent_size
        self.temp = temp
        # states encoder
        self.state_encoder = nn.Sequential(nn.Linear(self.obs_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, self.latent_size))
        # projection head
        self.proj = nn.Sequential(nn.Linear(2 * self.latent_size, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, self.latent_size))
        # skill encoder
        self.skill_encoder = nn.Sequential(nn.Linear(self.skill_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
                                        nn.Linear(hidden_dim, self.latent_size))
        
    def forward(self, states, next_states ,skills):
        # encode the states
        states = self.state_encoder(states)
        # encode the next states
        next_states = self.state_encoder(next_states)
        # project the skills
        projection = self.proj(torch.cat([state, next_states]), dim=1) # key
        # encode the skills 
        skills = self.skill_encoder(skills) # query
        # return the loss
        loss = self.info_nce_loss(projection, skills, self.temp)
        return loss
    
    def info_nce_loss(self, key, query, temp):
        temperature = temp
        eps = 1e-6 # for stability
        # normalization
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        cov = torch.mm(query,key.T) # (b,b)
        sim = torch.exp(cov / temperature) 
        neg = sim.sum(dim=-1) # (b,)
        row_sub = torch.Tensor(neg.shape).fill_(math.e**(1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability
        pos = torch.exp(torch.sum(query * key, dim=-1) / temperature) #(b,)
        loss = -torch.log(pos / (neg + eps)) #(b,)
        return loss
    
    
class Actor(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

        self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        policy_layers = []
        policy_layers += [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        # add additional hidden layer for pixels
        if obs_type == 'pixels':
            policy_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)
        

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist
    
class Critic(nn.Module):
    def __init__(self, cnn_encoder, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type
        
        self.cnn_encoder = cnn_encoder

        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            q_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, action, skill_dim=0, skill=None):
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                               dim=-1)
        # encode the images with the shared state encoder
        inpt = self.cnn_encoder(obs)
        if skill_dim > 0:
            obs = torch.cat([obs_encoded, skill], dim=1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2
    