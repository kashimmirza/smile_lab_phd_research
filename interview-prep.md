# SMILE Lab PhD Screening Preparation

## Core Evaluation Signals

1. Technical Depth
   - Production-grade code capability
   - Mathematical clarity without hand-waving

2. Research Alignment
   - Understanding of SMILE Lab mission: REVEAL++, microvascular dynamics, multimodal VLMs
   - Ability to add immediate value

3. Problem-Solving Ability
   - Handling edge cases, noisy medical data, failed training runs

---

## Strategic Answers for Top 5 Questions

### 1. Research Alignment Question

**Question:** "I saw your email mentioning 1D neural motion descriptors. How do you see this connecting to our work on REVEAL/REVEAL++ in the SMILE Lab?"

**Answer:**
- REVEAL++ connects static retinal microvascular structure with systemic health through vision-language modeling.
- Static imaging is only part of the picture; organ mechanics are dynamic in 3D+t.
- My approach compresses 3D cardiac motion fields into lightweight 1D directional cosine curves (`α_t`) using self-supervised registration.
- By concatenating these dynamic physical curves with static retinal OCTA embeddings inside REVEAL++, we create a cross-organ digital twin.
- This enriched representation captures microvascular density plus organ kinetics, supporting multimodal models like GatorTron without excessive compute.

### 2. Mathematics & Mechanics Question

**Question:** "Walk me through how you calculate the 1D motion descriptor `α_t`. Why directional cosine similarity instead of simple displacement magnitude?"

**Answer:**
- Displacement magnitude alone loses direction: outward expansion and inward contraction can appear identical.
- We compute a directional cosine similarity between local displacement `v_i` and a radial reference vector `w_i` toward the organ centroid:
  - `α_i = (v_i · w_i) / (||v_i|| ||w_i||)`
- The output is bounded in `[-1, 1]`.
- Negative values indicate inward contraction, positive values indicate outward relaxation.
- Averaging these values over the spatial region compresses many voxel vectors into a noise-robust 1D curve over time.

### 3. PyTorch & Architecture Question

**Question:** "How do you perform unsupervised image registration in PyTorch without ground-truth masks? How does the spatial transformer work?"

**Answer:**
- Use a 3D U-Net `f_θ(Imoving, Ifixed)` to output a continuous displacement field `ϕ_t`.
- Pass `Imoving` and `ϕ_t` into a Spatial Transformer Network (STN).
- The STN performs differentiable warping via `torch.nn.functional.grid_sample` with trilinear interpolation.
- Warped output `Ĩfixed` is compared to `Ifixed` using an image similarity loss like SSIM or local NCC.
- Add a smoothness regularizer on `∇ϕ_t`, such as the L2 norm of spatial gradients, to prevent unrealistic tissue tearing.

### 4. Data Engineering & Pathology Question

**Question:** "Medical data is notoriously noisy, multi-vendor, and full of artifacts. How do you handle domain shift and tissue abnormalities?"

**Answer:**
- Domain shift across vendors breaks supervised segmentation models.
- A self-supervised registration pipeline adapts directly to intensity distributions through SSIM-based similarity.
- For localized pathology (scar, fibrosis), uniform averaging degrades motion signals.
- Use voxel-wise weighting from T1/T2 relaxation maps or spatial confidence masks:
  - `α_t^tissue = (∑ M_i · T1_i · α_i) / (∑ M_i · T1_i)`
- This isolates diseased tissue kinetics from healthy myocardium.

### 5. PhD Fit & Motivation Question

**Question:** "Why do you want to pursue your PhD in the SMILE Lab at the University of Florida specifically?"

**Answer:**
- SMILE Lab uniquely bridges noninvasive microvascular imaging, foundation models, and clinical impact.
- UF Health clinical datasets and GatorTron infrastructure provide unmatched resources.
- I want to build noninvasive digital twins for early systemic disease detection.
- My background in PyTorch, spatial transformers, and vector algebra aligns directly with lab goals and grants.

---

## Quick Tips for the 15-Minute Call

- Keep answers under 90 seconds.
- Be precise and confident.
- Have your GitHub open for screen-sharing.
- Ask one high-level question at the end:
  - "Where do you see the biggest computational bottleneck right now in scaling REVEAL++ to multi-modal 3D+t clinical data?"

---

## Repo Goal

Use this repository as the primary place for research notes, interview preparation, and ongoing updates to support a fund application with Dr. Fang and the SMILE Lab.
