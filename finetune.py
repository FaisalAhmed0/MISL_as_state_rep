import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
from copy import deepcopy
import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import wandb 
import yaml

torch.backends.cudnn.benchmark = True


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg, encode_state=None):
    cfg.obs_type = obs_type
    print(f"current path {Path.cwd()}")
    if encode_state != "none":
        with open(f"/home/bethge/fmohamed65/MISL_as_state_rep/agent/{encode_state}.yaml") as f:
            file = yaml.safe_load(f)
        if encode_state != "none" and obs_type!="pixels":
            cfg.obs_shape =  obs_spec.shape
            cfg.meta_dim = 0
        else:
            cfg.obs_shape = obs_spec.shape
            cfg.meta_dim = int(file['skill_dim'])
    else:
        cfg.obs_shape = obs_spec.shape
        cfg.meta_dim = 0
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        print(f"obs type: {self.cfg.obs_type}")
        print(type(cfg))
        print(cfg["agent"])
        print(type(cfg["agent"]))
        # create logger
        if cfg.use_wandb:
            exp_name = cfg.data_folder
            hyperparams = {"lr": cfg["agent"]["lr"], "batch_size": cfg["agent"]["batch_size"], "tau": cfg["agent"]["critic_target_tau"], "feature_dim": cfg["agent"]["feature_dim"], "task": cfg.task, "seed": cfg.seed, "pretraining": cfg.pretrained_agent, "update_state_encoder": cfg.update_state_encoder, "obs_type":cfg.obs_type, "update_cnn_encoder": cfg.update_encoder, "state_encoder": cfg.state_encoder, "uid":cfg.uid, "skill_dim": cfg.entropy}
            print(f"exp_name:{exp_name}")
            print(f"hyper:{hyperparams}")
            wandb.init(project="vae_finetune",group=cfg.agent.name + '-ft',name=exp_name, config=hyperparams, settings=wandb.Settings(start_method='thread'))
            print("Connected to wandb")

        # create logger
        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)
        # create envs

        self.train_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)
        self.eval_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                 cfg.action_repeat, cfg.seed)
        # override the obervation shape if the state enocder will be used
        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent, 
                                encode_state = cfg.state_encoder)
        
            
        # check for using the state encoder
        if cfg.pretrained_agent != "none" and cfg.snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()['agent']
            print("pretrained agent is loaded")
            # load the cnn encoder incase of image observations
            if cfg.obs_type == "pixels":
                self.pixel_encoder = pretrained_agent.encoder
                print(f"pixel cnoder: {pretrained_agent.encoder}")
                self.agent.encoder = self.pixel_encoder
                # TODO: Initlize the encoder from simclr or vae
                # load the pre-trained encoder
                # copy the weights to the DrQ encoder
                if "ball_in_cup" in self.cfg.task:
                    domain = "ball_in_cup"
                else:
                    domain, _ = self.cfg.task.split('_', 1)
                if cfg.pretrained_encoder == "vae":
                    base_dir = "/mnt/qb/work/bethge/fmohamed65/MISL_as_state_rep/baselines/vae"
                    model_file = f"{base_dir}/{domain}/vae.ckpt"
                    self.agent.encoder.load_state_dict(torch.load(model_file))
                    print("VAE loaded successfully")
                elif cfg.pretrained_encoder == "cl":
                    base_dir = "/mnt/qb/work/bethge/fmohamed65/MISL_as_state_rep/baselines/contrastive"
                    model_file = f"{base_dir}/{domain}/cl.ckpt"
                    self.agent.encoder.load_state_dict(torch.load(model_file))
            # load the misl state encoder
            if cfg.state_encoder == "cic":
                self.state_encoder = pretrained_agent.cic.state_net
                print("cic agent has been assigned")
            elif cfg.state_encoder == "diayn":
                self.state_encoder = pretrained_agent.diayn.state_net    
            # load the state encoder on the DDPG agent
            if cfg.state_encoder != "none":
                self.agent.misl_state_encoder = self.state_encoder
                self.agent.finetune_state_encoder = cfg.update_state_encoder
                self.agent.misl_encoder_opt = None
            # create an optimizer for the state encoder if required
            if cfg.update_state_encoder:
                # extract the learning rate from the yaml file
                print(f"Current directory: {Path.cwd()}")
                with open("/home/bethge/fmohamed65/MISL_as_state_rep/agent/ddpg.yaml") as f:
                      file = yaml.safe_load(f)
                lr = float(file['lr'])
                self.agent.misl_encoder_opt = torch.optim.Adam(self.agent.misl_state_encoder.parameters(), lr=lr)
                self.agent.misl_state_encoder.train(self.agent.training)
            if cfg.update_encoder and cfg.obs_type == "pixels":
                with open("/home/bethge/fmohamed65/MISL_as_state_rep/agent/ddpg.yaml") as f:
                      file = yaml.safe_load(f)
                lr = float(file['lr'])
                self.agent.encoder_opt = torch.optim.Adam(self.agent.encoder.parameters(), lr=lr)
                self.agent.encoder.train(self.agent.training)

                      
        self.agent.train()
        self.agent.critic_target.train()
            
        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        current_time_step = deepcopy(time_step)
        meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('step', self.global_step)

                time_step = self.train_env.reset()
                current_time_step = deepcopy(time_step)
                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step.observation)

                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)

            if hasattr(self.agent, "regress_meta"):
                repeat = self.cfg.action_repeat
                every = self.agent.update_task_every_step // repeat
                init_step = self.agent.num_init_steps
                if self.global_step > (init_step // repeat) and self.global_step % every == 0:
                    meta = self.agent.regress_meta(self.replay_iter, self.global_step)

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            next_time_step = deepcopy(time_step)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1
            current_time_step = deepcopy(next_time_step)

    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        if "ball_in_cup" in self.cfg.task:
            domain = "ball_in_cup"
        else:
            domain, _ = self.cfg.task.split('_', 1)
        snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.pretrained_agent / self.cfg.experiment
        
        if self.cfg.snapshot_name != 'none':
            snapshot = snapshot_dir / f'snapshot_{self.cfg.snapshot_ts}_{self.cfg.snapshot_name}.pt'
        else:
            snapshot = snapshot_dir / f'snapshot_{self.cfg.snapshot_ts}_{self.cfg.experiment}.pt'
        print(f"snapshot path: {snapshot}")
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        return payload


@hydra.main(config_path='.', config_name='finetune')
def main(cfg):
    print(f"cfg: {cfg}")
    from finetune import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
