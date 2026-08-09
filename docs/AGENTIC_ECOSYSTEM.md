<!-- @format -->

# Agentic Ecosystem for SMILE Lab Research

This repository currently includes a research-ready PyTorch + MONAI + SimpleITK stack, plus W&B support for logging. It is intended to be extended into a full agentic research automation pipeline.

## Target Agentic Stack

1. **Sakana AI "The AI Scientist"**

   - Orchestrates paper writing, idea generation, literature search, and LaTeX reporting.
   - GitHub: https://github.com/SakanaAI/AI-Scientist

2. **CrewAI / Microsoft AutoGen**

   - Manages PyTorch training loops, MONAI transforms, dataset loading, and CUDA memory handling.
   - GitHub: https://github.com/crewAIInc/crewAI
   - GitHub: https://github.com/microsoft/autogen

3. **Weights & Biases / TensorBoard**

   - Logs ablation loss curves, Hessian metrics, and 1D motion `alpha_t` visualizations.

4. **MONAI + PyTorch + SimpleITK**
   - Executes 3D U-Net warping, spatial transformers, and directional vector mathematics.

## Current Status

- `requirements.txt` includes `monai`, `simpleitk`, `wandb`, `crewai`, and `autogen-agentchat`.
- The repository contains core modules for dataset loading, 3D registration, motion descriptor computation, and training scripts.
- The repository does not yet include a fully operational agent orchestration pipeline.

## Recommended Next Steps

1. Add a concrete AutoGen / CrewAI orchestration script to run `train_registration.py`, manage hyperparameters, and handle data preparation.
2. Add an agent script or notebook to generate LaTeX reports from experiment logs and evaluation metrics.
3. Add a `README` section describing how to bootstrap the agentic pipeline and which components are automated vs. manual.

## Implemented Orchestration

- `scripts/research_orchestrator.py` now performs dataset loading, model training, motion descriptor extraction, ablation summarization, and LaTeX report generation.
- `scripts/evaluate_ablation.py` is implemented to load experiment metrics and write a summary JSON report.
- AutoGen / CrewAI hooks remain available for future integration, while the main pipeline already runs end-to-end locally.

## Example Agentic Workflow

1. `Sakana AI` gathers relevant biomedical papers and generates a report outline.
2. `AutoGen` / `CrewAI` drives `scripts/train_registration.py` and `scripts/evaluate_ablation.py` across multiple settings.
3. `W&B` logs training curves and descriptor results.
4. The research pipeline produces figures, tables, and narrative outputs for the SMILE Lab.
