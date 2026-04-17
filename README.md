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
We present research results on the optimization of several fundamental structural objective functions. These include:

* Single-objective protein design, exemplified by optimizing the match_ss metric (e.g., from run XX_run1_0254_0003).
* Multi-objective design, utilizing a dual-objective combination of globularity and pLDDT.
* Multi-objective design, incorporating a triple-objective combination of hydrophobicity, globularity, and pLDDT.

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
Install pytroch, <a href="https://www.pyrosetta.org/">pyrosseta</a>. Then, run the following
```python
conda create -n MoMST python=3.9 
conda activate MoMST
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

Also, to optimize ```match_ss``` and ```crmsd```, go to the ```./datasets``` folder and download the protein examples as shown below. You can also use any PDB files.
```
python download_model_data.py
```
This code puts several pdb files into ```./datasets/AlphaFoldPDB/```.

### Example of Running the Code
Below is an explanation of the available options.

| Argument | Description |
|----------|------------|
| `--decoding` | Decoding method (`momst`, `SVDD_edit`, `SVDD`) |
| `--repeatnum` | Batch size |
| `--duplicate` | Number of andidates |
| `--metrics_name` | Reward functions |
| `--metrics_list` | Weights for rewards |
| `--proteinname` | Target PDB name |
| `--iteration` | Number of iterations |
| `--seq_length` | Protein length |

#### Single-Objective Protein Design
**1. Secondary Structure match**

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
**2. Hydrophobicity + Surface Exposure + pLDDT**

The hydrophobicity-surface exposure-pLDDT combination suits therapeutic protein design, ensuring high structural stability, solubility, and reduced aggregation-mediated immunogenic risks.
```
CUDA_VISIBLE_DEVICES=0 python refinement.py --decoding momst  --repeatnum 10 --duplicate 20  --metrics_name hydrophobic,surface_expose,plddt  --metrics_list 1,1,1 --iteration 20 --seq_length 150
```

## 🎓 Acknolwdgements
Our codebase is heavily based on <a href="https://github.com/masa-ue/ProDifEvo-Refinement?tab=readme-ov-file">RERD</a>, <a href="https://github.com/microsoft/evodiff">evodiff</a>, <a href="https://openfold.io/">openfold</a>, <a href="https://github.com/facebookresearch/esm">ESMfold</a>.
