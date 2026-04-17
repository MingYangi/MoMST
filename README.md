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
  <img src="https://github.com/MingYangi/MoMST/blob/main/medias/globularity%2Cplddt_1000.gif" width="20%">
</p>
<p align="center">
   <img src="https://github.com/MingYangi/MoMST/blob/main/medias/hydrophobic%2Cglobularity%2Cplddt_1000.gif
" width="35%">
</p>

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

</tr>
</table>

<table align="center" style="border-collapse: collapse; border: none;">
<tr>

<td align="center" width="50%" style="border: none; padding: 6px 20px;">
<img src="https://github.com/MingYangi/MoMST/blob/main/medias/globularity%2Cplddt_1000.gif?raw=true" width="92%">
<br><b>Globularity + pLDDT</b>
<br><span style="font-size: 12px;">Structural compactness</span>
</td>

<td align="center" width="50%" style="border: none; padding: 6px 20px;">
<img src="https://github.com/MingYangi/MoMST/blob/main/medias/hydrophobic%2Cglobularity%2Cplddt_1000.gif?raw=true" width="92%">
<br><b>Hydrophobic + Globularity + pLDDT</b>
<br><span style="font-size: 12px;">Multi-objective optimization</span>
</td>

</tr>
</table>

<div align="center">

<table style="border: none; border-collapse: collapse; border-spacing: 0;">
<tr>

<td style="border: none; padding: 0; text-align: center;">
<img src="https://github.com/MingYangi/MoMST/blob/main/medias/globularity%2Cplddt_1000.gif?raw=true" width="20%">
<br><b>Globularity + pLDDT</b><br>
Structural compactness
</td>

<td style="border: none; padding: 0; text-align: center;">
<img src="https://github.com/MingYangi/MoMST/blob/main/medias/hydrophobic%2Cglobularity%2Cplddt_1000.gif?raw=true" width="20%">
<br><b>Hydrophobic + Globularity + pLDDT</b><br>
Multi-objective optimization
</td>

</tr>
</table>

</div>


## 🚀 Quick Start
### Installation
Install pytroch, pyrosseta. Then, run the following
```python
conda create -n MoMST python=3.9 
conda activate MoMST
pip install torch torchvision torchaudio
pip install -r requirements.txt
