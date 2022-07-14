import math
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils
from agent.ddpg import DDPGAgent


	# An Encoder network for different MI critics
class Encoder(nn.Module):
    def __init__(self,  n_input, n_hiddens, n_latent, dropout=None):
        super().__init__()
        layers = []
        layers.append(nn.Linear(n_input, n_hiddens[0]))
        layers.append(nn.ReLU())
        if dropout:
              layers.append( nn.Dropout(dropout) )
                
        for i in range(len(n_hiddens)-1):
            layers.append( nn.Linear(n_hiddens[i], n_hiddens[i+1]) )
            layers.append( nn.ReLU() )
            if dropout:
                layers.append( nn.Dropout(dropout) )

        layers.append(nn.Linear(n_hiddens[-1], n_latent))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
class SeparableCritic(nn.Module):
    def __init__(self, state_dim, skill_dim, hidden_dims, latent_dim, temperature=1, dropout=None, device="cuda"):
        super().__init__()
        self.temp = temperature
        self.num_skills = skill_dim
        self.device = device
        # State encoder
        self.state_enc = Encoder(state_dim, hidden_dims, latent_dim, dropout=dropout)
        # Skill encoder
        self.skill_enc = Encoder(skill_dim, hidden_dims, latent_dim, dropout=dropout)
        
    def forward(self, x, y):
        x = F.normalize(self.state_enc(x), dim=-1) # shape (B * latent)
        y = F.normalize(self.skill_enc(y), dim=-1) # shape (B * latent)
        scores = torch.sum(x[:, None, :] * y[None, :, :], dim=-1) / self.temp #shape (B * B)
        info_nce = self.infoNCE_lower_bound(scores)
        return scores, info_nce
    
    # I_{NCE} lower bound, biased but low variance
    def infoNCE_lower_bound(self, scores, mean_reduction=False):
        batch_size = float(scores.shape[0])
        reduction = 'mean' if mean_reduction else 'none'
        nll = - nn.CrossEntropyLoss(reduction=reduction)(scores, target=torch.arange(int(batch_size)).to(self.device))
        mi = torch.log(torch.tensor(batch_size)) + nll
        return mi
    
    
class DIAYN(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim):
        super().__init__()
#         self.skill_pred_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
#                                             nn.ReLU(),
#                                             nn.Linear(hidden_dim, hidden_dim),
#                                             nn.ReLU(),
#                                             nn.Linear(hidden_dim, skill_dim))
        self.critic = SeparableCritic(obs_dim, skill_dim,[hidden_dim, hidden_dim], 64, temperature=0.5)

        self.apply(utils.weight_init)

    def forward(self, obs, skill):
#         skill_pred = self.skill_pred_net(obs)
        scores, info_nce = self.critic(obs, skill)
        return scores, info_nce


class DIAYNAgent(DDPGAgent):
    def __init__(self, update_skill_every_step, skill_dim, diayn_scale, **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.diayn_scale = diayn_scale
#         self.update_encoder = update_encoder
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim
        self.device = kwargs['device']

        # create actor and critic
        super().__init__(**kwargs)

        # create diayn
        self.diayn = DIAYN(self.obs_dim - self.skill_dim, self.skill_dim,
                           kwargs['hidden_dim']).to(kwargs['device'])

        # loss criterion
        self.diayn_criterion = nn.CrossEntropyLoss()
        # optimizers
        self.diayn_opt = torch.optim.Adam(self.diayn.parameters(), lr=self.lr)

        self.diayn.train()

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta(self):
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def update_diayn(self, skill, next_obs, step):
        metrics = dict()

        loss = self.compute_diayn_loss(next_obs.to(self.device), skill.to(self.device))

        self.diayn_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.diayn_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['diayn_loss'] = loss.item()
#             metrics['diayn_acc'] = df_accuracy

        return metrics

    def compute_intr_reward(self, skill, next_obs, step):
#         z_hat = torch.argmax(skill, dim=1)
#         d_pred = self.diayn(next_obs)
#         d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
#         _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
#         reward = d_pred_log_softmax[torch.arange(d_pred.shape[0]),
#                                     z_hat] - math.log(1 / self.skill_dim)
        _, reward = self.diayn(next_obs, skill)

        return reward * self.diayn_scale

    def compute_diayn_loss(self, next_state, skill):
        """
        DF Loss
        """
        scores, _ = self.diayn(next_state, skill)
        batch_size = next_state.shape[0]
        loss = nn.CrossEntropyLoss()(scores, target=torch.arange(int(batch_size)).to(self.device))
        return loss
#         z_hat = torch.argmax(skill, dim=1)
#         d_pred = self.diayn(next_state)
#         d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
#         _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
#         d_loss = self.diayn_criterion(d_pred, z_hat)
#         df_accuracy = torch.sum(
#             torch.eq(z_hat,
#                      pred_z.reshape(1,
#                                     list(
#                                         pred_z.size())[0])[0])).float() / list(
#                                             pred_z.size())[0]
#         return d_loss, df_accuracy

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
            batch, self.device)

        # augment and encode
        with torch.no_grad():
            obs = self.aug_and_encode(obs)
            next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            metrics.update(self.update_diayn(skill, next_obs, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, next_obs, step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

#         if not self.update_encoder:
#             obs = obs.detach()
#             next_obs = next_obs.detach()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
