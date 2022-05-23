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


class DISDAIN(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim, ensemble_size, lr):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.disc_models = []
        self.opts = []
        self.lr = lr
        self.skill_dim = skill_dim
        # loss criterion
        self.diayn_criterion = nn.CrossEntropyLoss()
        for _ in range(ensemble_size):
#             skill_pred_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
#                                             nn.ReLU(),
#                                             nn.Linear(hidden_dim, hidden_dim),
#                                             nn.ReLU(),
#                                             nn.Linear(hidden_dim, skill_dim))
            skill_pred_net = nn.Sequential(nn.Linear(obs_dim, skill_dim)).to("cuda")
            skill_pred_net.train()
            self.disc_models.append(skill_pred_net)
            self.opts.append(torch.optim.Adam(skill_pred_net.parameters(), lr=self.lr))

    def forward(self, obs):
        predictions = torch.zeros((obs.shape[0], self.skill_dim)).to('cuda')
        for i in range(self.ensemble_size):
            skill_pred = self.disc_models[i](obs)
            predictions += torch.softmax(skill_pred, dim=-1)
        return predictions/self.ensemble_size
    
    def loss(self, next_obs, skill, idx):
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.disc_models[idx](next_obs)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        d_loss = self.diayn_criterion(d_pred, z_hat)
        df_accuracy = torch.sum(
            torch.eq(z_hat,
                     pred_z.reshape(1,
                                    list(
                                        pred_z.size())[0])[0])).float() / list(
                                            pred_z.size())[0]
        return d_loss, df_accuracy
        
    
    # update each discriminator speratly
    def update(self, skill, next_obs):
        avg_loss = 0
        avg_accuracy = 0
        for idx in range(self.ensemble_size):
            loss, df_accuracy = self.loss(next_obs, skill, idx)
            self.opts[idx].zero_grad()
            loss.backward()
            self.opts[idx].step()
            avg_loss += loss.detach().item()
            avg_accuracy += df_accuracy.detach().item()
        avg_loss /= self.ensemble_size
        avg_accuracy /= self.ensemble_size
        return avg_loss, avg_accuracy

class DISDAINAgent(DDPGAgent):
    def __init__(self, update_skill_every_step, skill_dim, diayn_scale, ensemble_size, disdain_scale,**kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.diayn_scale = diayn_scale
        self.disdain_scale = disdain_scale
#         self.update_encoder = update_encoder
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim

        # create actor and critic
        super().__init__(**kwargs)

        # create diayn
        # 
        self.disdain = DISDAIN(self.obs_dim - self.skill_dim, self.skill_dim, 
                           kwargs['hidden_dim'], ensemble_size, self.lr)

        # loss criterion
#         self.diayn_criterion = nn.CrossEntropyLoss()
        # optimizers
#         self.diayn_opt = torch.optim.Adam(self.diayn.parameters(), lr=self.lr)

#         self.diayn.train()

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

        loss, df_accuracy = self.disdain.update(skill.to(self.device), next_obs.to(self.device))

#         self.diayn_opt.zero_grad()
#         if self.encoder_opt is not None:
#             self.encoder_opt.zero_grad(set_to_none=True)
#         loss.backward()
#         self.diayn_opt.step()
#         if self.encoder_opt is not None:
#             self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['disdain_loss'] = loss
            metrics['disdain_acc'] = df_accuracy

        return metrics
    
    @torch.no_grad()
    def compute_intr_reward(self, skill, next_obs, step):
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.disdain(next_obs.to(self.device))
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        reward = d_pred_log_softmax[torch.arange(d_pred.shape[0]),
                                    z_hat] - math.log(1 / self.skill_dim)
        reward = reward.reshape(-1, 1)
        
        r_disdain = self.compute_disdain_reward(next_obs.to(self.device), skill.to(self.device)) * self.disdain_scale
        
        return (reward + r_disdain) * self.diayn_scale

#     def compute_diayn_loss(self, next_state, skill):
#         """
#         DF Loss
#         """
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
    
    @torch.no_grad()
    def compute_disdain_reward(self, next_state, skill):
        # entropy of the mean discriminator
        prediction = self.disdain(next_state)
        mean_pred_probs = prediction
        mean_pred_logprobs = torch.log(prediction)
        term1 = -torch.sum(mean_pred_logprobs * mean_pred_probs, dim=-1)
        # mean of the entropies
        # calulate the entropy for each model
        sum_of_entropies = torch.zeros((next_state.shape[0], )).to(self.device)
        for model in self.disdain.disc_models:
            prediction = model(next_state)
            pred_probs = torch.softmax(prediction, dim=-1)
            pred_logprobs = torch.log_softmax(prediction, dim=-1)
            sum_of_entropies += ( -torch.sum(pred_probs * pred_logprobs, dim=-1) )
        term2 = sum_of_entropies/self.disdain.ensemble_size
        r = term1 - term2
        return r.reshape(-1, 1)
            
        
        

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

#             with torch.no_grad():
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
