import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optm as opt
from collections import OrderedDict

import utils

from agent.ddpg import DDPGAgent

from agent.cic import RMS
from agent.cic import APTArgs
from agent.cic import compute_apt_reward

from agent.models import Encoder # This cnn encoder already normalize the images in the forward function
from agent.models import SimCLR # simclr representation learner
from agent.models import CIC # CIC disctiminator
from agent.models import Actor
from agent.models import Critic
from agent.models import Transform


# training device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        
class CRLAgent:
    '''
    https://arxiv.org/abs/2105.01060
    '''
    def __init__(self, name,reward_free, obs_shape, action_shape,
                 device, lr, feature_dim, hidden_dim, critic_target_tau,
                 num_expl_steps, update_every_steps, stddev_schedule, nstep,
                 batch_size, stddev_clip, init_critic, use_wandb, update_encoder, temp, latent_size, objective, skill_dim=0, entropy_coef=0  ):
        self.temp = temp
        self.latent_size = latent_size
        self.objective = objective # Values: SimClr, CIC
        self.skill_dim = skill_dim
        self.entropy_coef = entropy_coef
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.feature_dim = feature_dim
        self.update_encoder = update_encoder
        self.batch_size = batch_size


        # Create the shared cnn encoder
        self.cnn_encoder = Encoder(obs_shape)
        
        # Initlize the representation learner depinding on the objective
        if objective == "SimClr":
            self.rep = SimCLR(temp, self.cnn_encoder, latent_size, hidden_dim)
            
        elif objective == "CIC":
            self.rep = CIC(self.cnn_encoder, obs_dim, skill_dim, latent_size, temp, hidden_dim)
        else:
            raise ValueError(f"Objective {objective} is not found")
        
        # create the optimizer
        self.rep_optim = opt.Adam(self.rep.parameters(), lr=lr)
        
        # set the observation dimenstion for the actor
        self.actor_obs_dim = self.cnn_encoder.repr_dim + skill_dim
        
        # create the actor and its optimizer
        self.actor = Actor('pixels', self.actor_obs_dim, self.action_dim,
                           feature_dim, hidden_dim).to(device)
        self.act_optim = opt.Adam(self.actor.parameters(), dim=1)
        
        # create the critic, target_critic and the optimizer
        self.critic = Critic(self.cnn_encoder, 'pixels', self.actor_obs_dim, self.action_dim,
                             feature_dim, hidden_dim).to(device)
        
        self.critic_target = Critic(self.cnn_encoder, 'pixels', self.actor_obs_dim, self.action_dim,
                             feature_dim, hidden_dim).to(device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.crt_optim = opt.Adam(self.critic_target.parameters(), lr=lr)
        
        # put everything in the training mode
        self.cnn_encoder.train()
        self.rep.train()
        self.actor.train()
        self.critic.train()
        
        
    def init_meta(self):
        if not self.reward_free:
            # selects mean skill of 0.5 (to select skill automatically use CEM or Grid Sweep
            # procedures described in the CIC paper)
            skill = np.ones(self.skill_dim).astype(np.float32) * 0.5
        else:
            skill = np.random.uniform(0,1,self.skill_dim).astype(np.float32)
        meta = OrderedDict()
        meta['skill'] = skill
        return meta
        
    def act(self, obs, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        if self.update_encoder:
            h = self.encoder(obs)
        else:
            h = self.encoder(obs).detach()
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        #assert obs.shape[-1] == self.obs_shape[-1]
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(inpt, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]
    
    
    def update_rep(self, obs, next_obs=None, skills=None):
        #TODO: add image augmentation
        if self.objective == "SimClr":
            # augmentation one
            obs1 = self.rep.transform(obs.copy())
            obs2 = self.rep.transform(obs.copy())
            loss = self.rep(obs1, obs2)
            self.rep_optim.zero_grad()
            loss.backward()
            self.rep_optim.step()
            return obs1.detach(), obs2.detach()
        elif self.objective == "CIC":
            loss = self.repd(obs, next_obs ,skills)
            self.rep_optim.zero_grad()
            loss.backward()
            self.rep_optim.step()
            
            
    
    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        # Calculate the target values
        with torch.no_grad():
            # If the observation are not the latent states pass them through the state encoder here.
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            neg_entropy = torch.sum(dist.log_prob(next_action), dim=1).reshape(-1, 1)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.entropy_coef * neg_entropy
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
#         if self.encoder_opt is not None:
#             self.encoder_opt.zero_grad(set_to_none=True)
#         if self.finetune_state_encoder:
#             self.misl_encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
#         if self.encoder_opt is not None:
#             self.encoder_opt.step()
#         if self.finetune_state_encoder:
#             self.misl_encoder_opt.step()
        return metrics

    def update_actor(self, obs, step):
        metrics = dict()
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()
        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics
    
    # CIC reward (entorpy term)
    @torch.no_grad()
    def compute_apt_reward(self, obs, next_obs):
        args = APTArgs()
        source = self.rep.state_encoder(obs)
        target = self.rep.state_encoder(next_obs)
        reward = compute_apt_reward(source, target, args) # (b,)
        return reward.unsqueeze(-1) # (b,1)
    
    @torch.no_grad()
    def compute_intr_reward(self, obs, obs2=None, next_obs=None):
        if self.objective == "SimClr":
            reward = self.rep.info_nce_reward(obs, obs2)
            return reward
        elif self.objective == "CIC":
            reward = self.compute_apt_reward(obs, next_obs)
        return reward
    
    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)
    
    def update(self, replay_iter, step):
        metrics = dict()
        
        if step % self.update_every_steps != 0:
            return metrics
        
        batch = next(replay_iter)
        self.batch = batch

        obs, action, reward, discount, next_obs, skill = utils.to_torch(
            batch, self.device)
        
        # update the representation learner
        if self.objective == "SimClr":
            obs1, obs2 = self.update_rep(obs)
        elif self.objective == "CIC":
            self.update_rep(obs, next_obs, skill)
        
        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encoder(next_obs)
        
        
        if self.reward_free:
            if self.objective == "SimClr":
            intr_reward = self.compute_intr_reward(obs=obs1, obs2=obs2)
        elif self.objective == "CIC":
            intr_reward = self.compute_intr_reward(obs=obs, obs2=obs2)
            
            reward = intr_reward.reshape(-1, 1)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

           

        return metrics
        
        
        
        
        
        
        
        
        
        
        