<!-- @format -->

# Aims and Scope for SMILE Lab Research

## Project Vision

This repository supports a research pipeline for self-supervised motion descriptor extraction, pathology-weighted multi-parametric integration, and multimodal REVEAL++ alignment.

## REVEAL / REVEAL++ Alignment

The proposed work is positioned as a bridge between physical cardiac dynamics and retinal microvascular biomarkers. The 3D+t motion descriptor pipeline produces compact, interpretable signals from cine CMR data, while the multimodal fusion scaffolding is designed to complement retinal OCTA and text-based features within REVEAL++.

This framing demonstrates how the project can contribute to a future foundation-model workflow in which dynamic organ mechanics and static microvascular structure are jointly represented for earlier, noninvasive systemic disease detection.

## Aim 1: Baseline Motion Extraction

- Train a deformable 3D U-Net to estimate motion fields from cine CMR scans.
- Use a Spatial Transformer Network (STN) to warp moving frames into fixed reference space.
- Compute a self-supervised loss: SSIM similarity plus spatial smoothness.
- Derive a 1D motion descriptor curve `α_t` from the resulting displacement field.

## Aim 2: Pathology-Weighted Multi-Parametric Integration

- Integrate voxel-wise T1/T2 map weighting into the motion descriptor.
- Evaluate whether pathology-weighted descriptors better capture scar and fibrosis dynamics.
- Compare weighted and unweighted motion signals across tissue regions.

## Aim 3: Systematic Ablation and Multimodal Fusion

- Run ablation experiments across static OCTA, dynamic motion, and multimodal fusion.
- Log results with W&B or TensorBoard for reproducibility.
- Demonstrate how `α_t^{tissue}` can augment REVEAL++ and cross-organ modeling.

## Expected Deliverables

- Reproducible PyTorch pipeline for 3D registration and descriptor extraction.
- Publication-ready analytics including Bland-Altman plots, confidence intervals, and p-values.
- Research documentation aligned with SMILE Lab's REVEAL++ mission.
