# MoMST: Multi-objective Protein Design via Memory-aware Test-time Scaling in Diffusion Models

<p align="center">
  :sparkles: Official implementation of our paper
</p>
<p align="center">
  📰 paper | 🔗 <a href="https://github.com/MingYangi/MoMST/tree/main">code</a>
</p>

## 🧠 Overview
This repository implements **MOMST**, a framework for multi-objective protein sequence design. This framework alternates between noising and memory-guided denoising in diffusion models. By combining self-contrastive learning to extract residue-level preferences from historical trajectories with inference-time Pareto alignment, MOMST effectively balances conflicting functional rewards while strictly preserving the pre-trained model's sequence naturalness.

## : Generated Proteins
### The Presentation of globularity and pLDDT result
<p align="left">
  <img src="https://github.com/MingYangi/MoMST/blob/main/medias/globularity%2Cplddt_1000.gif" width="35%">
  <img src="https://github.com/MingYangi/MoMST/blob/main/medias/hydrophobic%2Cglobularity%2Cplddt_1000.gif
" width="35%">
</p>

## 🚀 Quick Start
### Installation
Install pytroch, pyrosseta. Then, run the following
```python
conda create -n MoMST python=3.9 
conda activate MoMST
pip install torch torchvision torchaudio
pip install -r requirements.txt
