<!-- @format -->

# Medical Image  Lab PhD Research Prep
<img width="1666" height="1192" alt="image" src="https://github.com/user-attachments/assets/527fa07d-7d26-420d-9400-54e6e3d26ee7" />


This repository is a research-ready scaffold for SMILE Lab PhD preparation, agentic medical imaging experimentation, and AI-assisted scientific workflow development. It is organized to support both interview preparation and technical research communication for Dr. Ruogu Fang and the SMILE Lab.

## Objective

Build and maintain a clear, reproducible research repo for:

- interview preparation and motivation framing
- research alignment with SMILE Lab priorities
- agentic experimentation on multimodal medical imaging data
- technical storytelling around motion descriptors, registration, and fusion methods

## What this repo supports

- multimodal 3D MRI and OCTA/fundus preprocessing workflows
- registration and deformation modeling experiments
- ablation studies and metric tracking
- reproducible reporting and visualization outputs
- AI-assisted research planning with LLM-compatible environment hooks

## Repository contents

- [configs/default_config.yaml](configs/default_config.yaml) — dataset paths, training hyperparameters, and logging settings
- [data/dataset_mri.py](data/dataset_mri.py) — 3D MRI dataset loader using NiBabel and SimpleITK
- [data/dataset_octa.py](data/dataset_octa.py) — OCTA/fundus feature loader for multimodal experiments
- [data/transforms.py](data/transforms.py) — MONAI preprocessing and augmentation pipeline definitions
- [models/unet3d_registration.py](models/unet3d_registration.py) — 3D U-Net registration model for deformable motion field estimation
- [models/spatial_transformer.py](models/spatial_transformer.py) — differentiable STN warping via PyTorch grid sampling
- [models/motion_engine.py](models/motion_engine.py) — directional cosine similarity and motion descriptor math
- [models/multimodal_fusion.py](models/multimodal_fusion.py) — modular fusion head for OCTA, motion, and text features
- [losses/ssim_3d.py](losses/ssim_3d.py) — self-supervised image similarity loss prototype
- [losses/smoothness.py](losses/smoothness.py) — spatial smoothness regularizer for deformation fields
- [losses/contrastive.py](losses/contrastive.py) — contrastive loss scaffold for cross-modal alignment experiments
- [scripts/train_registration.py](scripts/train_registration.py) — training script for unsupervised 3D registration
- [scripts/extract_descriptors.py](scripts/extract_descriptors.py) — extract 1D motion descriptor curves from learned flows
- [scripts/evaluate_ablation.py](scripts/evaluate_ablation.py) — evaluate ablation metrics and summarize results
- [scripts/research_orchestrator.py](scripts/research_orchestrator.py) — orchestrate dataset prep, training, descriptor extraction, ablation, and report generation
- [notebooks/01_visualization.ipynb](notebooks/01_visualization.ipynb) — example notebook for visualizing warps and motion curves
- [docs/AIMS_AND_SCOPE.md](docs/AIMS_AND_SCOPE.md) — research plan aligned with SMILE Lab aims
- [interview-prep.md](interview-prep.md) — tailored responses to likely screening questions
- [requirements.txt](requirements.txt) — reproducible package environment for the project

## Research focus

1. **Agentic data prep and harmonization**
   - multi-site 3D medical imaging pipelines using SimpleITK, MONAI, and PyTorch
2. **Agentic model exploration and ablation**
   - 3D U-Net registration, STN warping, pathology-weighted descriptors, and loss-variant comparisons
3. **Agentic mathematical validation and physics consistency**
   - directional cosine similarity, divergence checks, and zero-crossing keyframe analysis
4. **Agentic statistical benchmarking and visualization**
   - outputs for publication-ready tables, confidence intervals, and agreement plots

## Quick start

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Windows cmd
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
cp .env.example .env
python scripts/research_orchestrator.py --config configs/default_config.yaml
```

## Environment variables

For optional AI-assisted experimentation or model-backed workflows, configure the following values in the local environment or the copied .env file:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `S2_API_KEY`
- `OPENALEX_MAIL_ADDRESS`
- `WANDB_API_KEY`

## Experimental analysis design

The current scaffold is intentionally simple, but it is designed to support a meaningful ablation study around image perturbations and multimodal reasoning. A useful analysis plan is:

- rotate images by small angles such as $0^\\circ, 5^\\circ, 10^\\circ, 15^\\circ, 20^\\circ$
- translate images by a few pixels to test spatial sensitivity
- crop the image with ratios such as $1.0, 0.95, 0.90, 0.85$ and resize back to the original shape
- add Gaussian noise with standard deviations such as $0.0, 0.01, 0.03, 0.05$
- vary brightness slightly to mimic scanner or acquisition differences
# SMILE Lab PhD Research Portfolio

## 📄 Key Publications & Technical Summaries

* 📕 **[ConvNeXt-CBAM Technical Brief (PDF)](./notebooks/ConvNeXt-CBAM%20FOR%20BREAST%20ULTRASOUND.pdf)**  
  *Evaluating Lightweight Dual-Attention Architectures for Breast Ultrasound Classification.*
* 📘 **[PPV & Bayes Derivation Handout (PDF)](./notebooks/truefalsepositiveandtrupositive.pdf)**  
  *Step-by-step mathematical derivation of Positive Predictive Value using Beamer slides.*

---

## 🔬 Coursera / Medical AI Lab Scripts

| Notebook Script | Topic |
| :--- | :--- |
| [`C1_W1_Lab_1`](./notebooks/C1_W1_Lab_1_data_exploration_and_image_preprocessing%20(1).py) | Image Preprocessing & Data Exploration |
| [`C1_W1_Lab_2`](./notebooks/C1_W1_Lab_2_counting_labels_and_weighted_loss_function.py) | Class Imbalance & Weighted Loss Functions |
| [`C1_W1_Lab_3`](./notebooks/C1_W1_Lab_3_densenet.py) | DenseNet Architecture Implementation |
| [`C1_W1_Lab_4`](./notebooks/C1_W1_Lab_4_patient_overlap_and_data_leakage.py) | Patient Overlap & Preventing Data Leakage |
These experiments help measure whether the model is robust to realistic acquisition shifts, which is especially important for medical imaging.

## Alignment with SMILE Lab

This repository is designed to communicate the following to Dr. Ruogu Fang and the SMILE Lab:

- production-grade PyTorch engineering
- explicit mathematical motion descriptor design
- domain-aware medical imaging workflows
- multimodal fusion readiness for REVEAL++
- reproducible experimentation, ablation logging, and thoughtful scientific framing
