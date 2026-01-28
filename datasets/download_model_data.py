import wandb
import os
run = wandb.init()


artifact = run.use_artifact("fderc_diffusion/Inverse_PF/AlphaFoldPDB:v0")
dir = artifact.download()
os.system('tar -xvzf artifacts/AlphaFoldPDB:v0/AlphaFoldPDB.tar.gz')
# https://wandb.ai/fderc_diffusion/Inverse_PF/artifacts/dataset/AlphaFoldPDB/v0/files

wandb.finish()
