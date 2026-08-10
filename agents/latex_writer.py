"""
agents/latex_writer.py
------------------------
Uses Claude to draft LaTeX prose and equations for the research report,
grounded in the actual loss functions and formulas implemented in this
repo (not generic boilerplate). Can write a single section on instruction,
or assemble a full journal-style paper, and optionally compiles it to PDF
if pdflatex/tectonic is available on PATH.
"""

import os
import shutil
import subprocess

MODEL = "claude-sonnet-5"

# Grounding context: the real math already implemented in this repo, so the
# model writes equations that match the code instead of inventing generic ones.
KNOWN_METHODS_CONTEXT = """
This paper describes an unsupervised 3D image registration and multimodal
motion-descriptor pipeline. The implemented components are:

1. Registration network: a 3D U-Net (models/unet3d_registration.py) predicts
   a dense deformation field phi from an input volume I.
2. Warping: a differentiable spatial transformer (models/spatial_transformer.py)
   warps I by phi via trilinear grid sampling to produce I_warped.
3. Losses (losses/ssim_3d.py, losses/smoothness.py):
   - L_ssim = 1 - SSIM(I_warped, I), a 3D structural similarity loss.
   - L_smooth = sum over voxels of the squared gradient magnitude of phi,
     a smoothness/regularization term on the deformation field.
   - Total loss: L = w_ssim * L_ssim + w_smooth * L_smooth
     (weights set in config.loss.ssim_weight / smoothness_weight).
4. Motion descriptor (models/motion_engine.py):
   - A radial reference field r is built from a centroid.
   - Directional cosine similarity alpha(x) = (phi(x) . r(x)) / (|phi(x)| |r(x)|)
     is computed per voxel.
   - The scalar descriptor is the mean of alpha(x) over the volume.
"""


def _get_client():
    import anthropic
    return anthropic.Anthropic()


def write_latex_section(instruction: str, session, extra_context: str = "") -> str:
    """
    Ask Claude to draft ONE LaTeX section (methods, results, discussion, a
    single equation, etc.), grounded in the actual pipeline math and, if
    available, this session's real metrics (loss history, ablation results,
    descriptor stats) -- so numbers cited are real, not fabricated.
    """
    client = _get_client()

    metrics_summary = ""
    if session.metrics:
        metrics_summary = (
            f"Actual results from this run: final training loss = "
            f"{session.metrics.get('final_loss')}, mean alpha descriptor = "
            f"{session.metrics.get('descriptor_mean_alpha')}. "
            f"Ablation results: {session.metrics.get('ablation_results')}."
        )

    system = (
        "You are a scientific writing assistant for a biomedical imaging paper. "
        "Write ONLY valid LaTeX for the requested section -- no markdown, no code fences, "
        "no explanation outside the LaTeX itself. Use \\subsection*{} for the heading. "
        "Use proper equation environments (equation, align) for all math, with \\label{} "
        "on each numbered equation. Ground every equation in the method description given "
        "below -- do not invent formulas that aren't implied by it. If real metric values "
        "are provided, cite them exactly; do not fabricate numbers or results."
    )

    user_msg = (
        f"Method description (ground truth, from the actual codebase):\n{KNOWN_METHODS_CONTEXT}\n\n"
        f"{extra_context}\n{metrics_summary}\n\n"
        f"Instruction for this section: {instruction}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(b.text for b in response.content if b.type == "text")


JOURNAL_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}
\title{%s}
\author{%s}
\date{\today}
\begin{document}
\maketitle
"""


def build_full_paper(session, title: str, author: str, section_instructions: list,
                      output_path: str, compile_pdf: bool = False) -> dict:
    """
    section_instructions: list of (section_name, instruction) pairs, e.g.
        [("Abstract", "..."), ("Introduction", "..."), ("Methods", "..."),
         ("Results", "..."), ("Discussion", "...")]
    Writes output_path (.tex). If compile_pdf=True and pdflatex/tectonic is
    on PATH, also compiles it and returns the PDF path.
    """
    parts = [JOURNAL_PREAMBLE % (title, author)]
    for name, instruction in section_instructions:
        section_tex = write_latex_section(instruction, session, extra_context=f"Section: {name}")
        parts.append(section_tex)
        parts.append("\n\n")
    parts.append(r"\end{document}")

    tex = "\n".join(parts)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(tex)

    result = {"tex_path": output_path, "pdf_path": None, "compiled": False}

    if compile_pdf:
        engine = shutil.which("pdflatex") or shutil.which("tectonic")
        if engine is None:
            result["compile_note"] = (
                "No LaTeX engine (pdflatex/tectonic) found on PATH; .tex was written but not compiled. "
                "Install texlive (or tectonic) to enable PDF compilation."
            )
        else:
            out_dir = os.path.dirname(output_path) or "."
            try:
                if "tectonic" in engine:
                    subprocess.run([engine, output_path], cwd=out_dir, check=True, capture_output=True)
                else:
                    subprocess.run(
                        [engine, "-interaction=nonstopmode", "-output-directory", out_dir, output_path],
                        check=True, capture_output=True,
                    )
                pdf_path = output_path.replace(".tex", ".pdf")
                if os.path.exists(pdf_path):
                    result["pdf_path"] = pdf_path
                    result["compiled"] = True
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode(errors="ignore")[-500:] if e.stderr else str(e)
                result["compile_note"] = f"Compilation failed: {stderr}"

    return result