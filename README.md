# MoMST: Multi-objective Protein Design via Memory-aware Test-time Scaling in Diffusion Models

<p align="center">
  :sparkles: Official implementation of our paper
</p>
<p align="center">
  📰 paper | 🔗 <a href="https://github.com/MingYangi/MoMST/tree/main">code</a>
</p>

## 🧠 Overview
This repository implements **MOMST**, a framework for multi-objective protein sequence design. This framework alternates between noising and memory-guided denoising in diffusion models. By combining self-contrastive learning to extract residue-level preferences from historical trajectories with inference-time Pareto alignment, MOMST effectively balances conflicting functional rewards while strictly preserving the pre-trained model's sequence naturalness.

## 🧬 Generated Proteins
### The Presentation of Result
<!-- <div align="center" style="display:flex; gap:20px; justify-content:center;">
  <div>
    <img src="https://github.com/MingYangi/MoMST/blob/main/medias/globularity%2Cplddt_1000.gif?raw=true" width="20%">
    <p align="center"><b>Globularity + pLDDT</b><br>Structural compactness and confidence score visualization</p>
  </div>

  <div>
    <img src="https://github.com/MingYangi/MoMST/blob/main/medias/hydrophobic%2Cglobularity%2Cplddt_1000.gif?raw=true" width="20%">
    <p align="center"><b>Hydrophobicity + Globularity + pLDDT</b><br>Multi-objective protein property optimization result</p>
  </div>

</div>
<div align="center"> -->

<div align="center">
<img src="https://github.com/MingYangi/MoMST/blob/main/medias/XX_run1_0254_0003.png" width="20%">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://github.com/MingYangi/MoMST/blob/main/medias/globularity%2Cplddt.gif?raw=true" width="20%">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://github.com/MingYangi/MoMST/blob/main/medias/hydrophobic%2Cglobularity%2Cplddt.gif?raw=true" width="20%">

ss-match (XX_run1_0254_0003)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Globularity + pLDDT
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Hydrophobic + Globularity + pLDDT
</div>


## 🚀 Quick Start
### Installation
Install pytroch, pyrosseta. Then, run the following
```python
conda create -n MoMST python=3.9 
conda activate MoMST
pip install torch torchvision torchaudio
pip install -r requirements.txt
```
### Running the Code
#### Single-Objective Protein Design
**1. SS (secondary structure) match**

Design a sequence that folds into a target secondary structure.
```
CUDA_VISIBLE_DEVICES=0 python refinement.py --decoding momst  --repeatnum 10 --duplicate 20  --metrics_name match_ss  --metrics_list 1 --proteinname XX_run1_0254_0003 --iteration 30
```
**2. cRMSD**

Design a sequence that folds into a target structure based on cRMSD.
```
CUDA_VISIBLE_DEVICES=0 python refinement.py --decoding momst  --repeatnum 20 --duplicate 20  --metrics_name crmsd  --metrics_list 1 --proteinname 5KPH --iteration 40
```
#### Multi-Objective Protein Design
**1. Globularity + pLDDT**

The globularity-pLDDT combination provides structural confidence in a compact sphere for stable scaffold design.
```
CUDA_VISIBLE_DEVICES=0 python refinement.py --decoding momst  --repeatnum 10 --duplicate 20  --metrics_name globularity,plddt  --metrics_list 1,1 --iteration 20 --seq_length 150
```

## Acknolwdgements
Our codebase is heavily based on <a href="https://github.com/masa-ue/ProDifEvo-Refinement?tab=readme-ov-file">RERD</a>, <a href="https://github.com/microsoft/evodiff">evodiff</a>, <a href="https://openfold.io/">openfold</a>, <a href="https://github.com/facebookresearch/esm">ESMfold</a>.
