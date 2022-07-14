# Mutual Informationn Skill As State Representation Learning
## Getting Started
We assume that you have an access to GPU that can run CUDA 10.4 or above.
You can create a conda environment and install the dependencies with the following command.
```bash
conda env create -f conda_env.yml
```
After that you can activate the environment
```bash
conda activate urlb
```
## Pretraining
Example: ``` python pretrain.py agent=cic domain=walker experiment=NAME ```.

Currently, we are experimtning with CIC and DIAYN.

## Training for a task using the pretrained representations
Example ```python finetune.py pretrained_agent=cic agent=ddpg experiment=NAME snapshot_ts=1000000 task=walker_walk```

## TODO
1. Add Autoencoder-based baseline.
2. Add DIAYN (additional skill learning baseline).
3. Run the state-based learning experiment with lower state dimension.

