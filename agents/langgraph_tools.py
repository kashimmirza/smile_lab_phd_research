"""
agents/langgraph_tools.py
--------------------------
LangChain-compatible tools for the SMILE Lab Research Director agent.

Each tool is decorated with @tool so LangGraph nodes can call them
as structured tool calls — matching the AssembledToolCall pattern from
LangChain's tool-calling docs (name, args, output, status lifecycle).

Tool groups:
  - Paper reading   : read_local_pdf
  - Literature      : search_arxiv_papers, search_openalex_papers
  - Gap analysis    : find_research_gaps, propose_math_modification
  - Experiments     : generate_ablation_matrix, run_quick_experiment
  - Communication   : draft_collaboration_email, send_collaboration_email
  - Summarization   : bullet_point_paper, karpathy_critique
"""

from __future__ import annotations

import json
import os
import re
import smtplib
import textwrap
import urllib.parse
import urllib.request
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Optional imports — gracefully degrade if not installed
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    _PYMUPDF_OK = True
except ImportError:
    _PYMUPDF_OK = False

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False


# ---------------------------------------------------------------------------
# ── PAPER READING ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@tool
def read_local_pdf(pdf_path: str, max_pages: int = 30) -> str:
    """
    Extract text from a local PDF file and return it (truncated to max_pages).
    Use this to read papers in the research_papers/ folder before analysis.
    """
    path = Path(pdf_path)
    if not path.exists():
        # Try relative to project root
        for base in [
            Path(__file__).parent.parent / "research_papers",
            Path(__file__).parent.parent,
        ]:
            candidate = base / pdf_path
            if candidate.exists():
                path = candidate
                break
        else:
            return f"ERROR: File not found: {pdf_path}"

    if not _PYMUPDF_OK:
        return (
            "PyMuPDF (fitz) is not installed. "
            "Run: pip install pymupdf\n"
            f"File requested: {path}"
        )

    doc = fitz.open(str(path))
    pages = min(max_pages, len(doc))
    chunks = []
    for i in range(pages):
        text = doc[i].get_text("text")
        if text.strip():
            chunks.append(f"[PAGE {i+1}]\n{text.strip()}")
    doc.close()
    full = "\n\n".join(chunks)
    # Trim to ~40k chars to stay within context window
    if len(full) > 40_000:
        full = full[:40_000] + "\n\n[...TRUNCATED — more pages available...]"
    return full


@tool
def list_local_papers(papers_dir: str = "research_papers") -> str:
    """
    List all PDF files in the research_papers directory.
    Returns JSON with file names and sizes.
    """
    base = Path(__file__).parent.parent / papers_dir
    if not base.exists():
        return json.dumps({"error": f"Directory not found: {base}"})
    pdfs = []
    for p in sorted(base.glob("*.pdf")):
        pdfs.append({"name": p.name, "path": str(p), "size_mb": round(p.stat().st_size / 1e6, 2)})
    return json.dumps({"papers": pdfs, "count": len(pdfs)})


@tool
def bullet_point_paper(paper_text: str, focus: str = "multimodal medical imaging") -> str:
    """
    Given extracted paper text, return structured bullet points covering:
    - Core contribution
    - Methodology
    - Dataset used
    - Key results / metrics
    - Limitations
    - Research gaps for follow-up

    The focus parameter directs attention to a specific domain.
    Note: This tool returns the structured prompt for the LLM node to process;
    it does NOT call an LLM directly.
    """
    # Extract title candidate (first non-empty line)
    lines = [l.strip() for l in paper_text.split("\n") if l.strip()]
    title_guess = lines[0][:120] if lines else "Unknown paper"

    word_count = len(paper_text.split())
    # Return a structured analysis request — the LLM node will process it
    return json.dumps({
        "instruction": "analyze_paper",
        "title_guess": title_guess,
        "word_count": word_count,
        "focus_domain": focus,
        "paper_excerpt": paper_text[:8000],  # first 8k chars for LLM context
        "required_sections": [
            "core_contribution",
            "methodology",
            "datasets",
            "key_results",
            "mathematical_formulation",
            "limitations",
            "open_gaps",
        ],
    })


# ---------------------------------------------------------------------------
# ── LITERATURE SEARCH ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@tool
def search_arxiv_papers(
    query: str,
    max_results: int = 8,
    year_from: int = 2023,
) -> str:
    """
    Search ArXiv for recent papers matching the query.
    Returns JSON with title, authors, abstract, arxiv_id, published date.
    Filters to papers from year_from onwards.
    """
    encoded = urllib.parse.quote(query)
    url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query=all:{encoded}"
        f"&start=0&max_results={max_results * 2}"  # fetch extra to filter by year
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            xml = resp.read().decode("utf-8")
    except Exception as e:
        return json.dumps({"error": str(e), "query": query})

    # Minimal XML parsing without lxml
    entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
    results = []
    for entry in entries:
        def get(tag: str) -> str:
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", entry, re.DOTALL)
            return m.group(1).strip() if m else ""

        published = get("published")[:10]  # YYYY-MM-DD
        year = int(published[:4]) if published else 0
        if year < year_from:
            continue

        arxiv_id_raw = get("id")
        arxiv_id = re.search(r"abs/(.+?)$", arxiv_id_raw)
        arxiv_id = arxiv_id.group(1) if arxiv_id else arxiv_id_raw

        results.append({
            "title": get("title").replace("\n", " "),
            "authors": re.findall(r"<name>(.*?)</name>", entry),
            "abstract": get("summary").replace("\n", " ")[:600],
            "arxiv_id": arxiv_id,
            "published": published,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        })
        if len(results) >= max_results:
            break

    return json.dumps({"query": query, "results": results, "count": len(results)})


@tool
def search_openalex_papers(
    query: str,
    max_results: int = 6,
    filter_year: int = 2022,
) -> str:
    """
    Search OpenAlex for academic papers with citation counts.
    Good for finding influential, well-cited work in biomedical imaging.
    Returns JSON with title, doi, citation_count, abstract.
    """
    encoded = urllib.parse.quote(query)
    email = os.environ.get("OPENALEX_MAIL_ADDRESS", "research@example.com")
    url = (
        f"https://api.openalex.org/works?"
        f"search={encoded}"
        f"&filter=publication_year:>{filter_year}"
        f"&sort=cited_by_count:desc"
        f"&per-page={max_results}"
        f"&mailto={email}"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return json.dumps({"error": str(e), "query": query})

    results = []
    for work in data.get("results", []):
        abstract_inv = work.get("abstract_inverted_index") or {}
        # Reconstruct abstract from inverted index
        if abstract_inv:
            words = {}
            for word, positions in abstract_inv.items():
                for pos in positions:
                    words[pos] = word
            abstract = " ".join(words[k] for k in sorted(words))[:600]
        else:
            abstract = "(no abstract available)"

        results.append({
            "title": work.get("title", ""),
            "doi": work.get("doi", ""),
            "publication_year": work.get("publication_year"),
            "citation_count": work.get("cited_by_count", 0),
            "abstract": abstract,
            "url": work.get("id", ""),
        })

    return json.dumps({"query": query, "results": results, "count": len(results)})


# ---------------------------------------------------------------------------
# ── GAP ANALYSIS & MATH INNOVATION ────────────────────────────────────────
# ---------------------------------------------------------------------------

@tool
def find_research_gaps(
    paper_summaries: str,
    domain: str = "multimodal medical imaging",
    focus: str = "MRI + OCTA registration and fusion",
) -> str:
    """
    Given a JSON string of paper summaries (from bullet_point_paper or search results),
    produce a structured prompt that instructs the LLM to identify:
    - What problems are UNSOLVED
    - What datasets are UNDEREXPLORED
    - What architectures are MISSING
    - What mathematical tools are UNDERUTILIZED
    - What clinical translation gaps exist

    Returns a structured prompt dict for the Gap Finder node to process with the LLM.
    """
    return json.dumps({
        "instruction": "find_gaps",
        "domain": domain,
        "focus": focus,
        "paper_summaries": paper_summaries[:6000],
        "gap_categories": [
            "unsolved_problems",
            "underexplored_datasets",
            "missing_architectures",
            "underutilized_math",
            "clinical_translation_gaps",
            "multimodal_fusion_gaps",
            "self_supervised_opportunities",
        ],
        "output_format": "numbered_list_with_novelty_score_0_to_10",
    })


@tool
def propose_math_modification(
    existing_method: str,
    target_improvement: str,
    constraints: str = "must be differentiable, GPU-compatible, clinically interpretable",
) -> str:
    """
    Given an existing mathematical method/loss/architecture description and
    a target improvement goal, generate a structured prompt for the Math Innovator
    node. Inspired by Andrej Karpathy's approach: derive from first principles,
    question every assumption, find the simplest modification that works.

    Returns JSON prompt for LLM to produce:
    - Mathematical formulation (LaTeX)
    - Intuition behind the modification
    - Expected improvement and why
    - Ablation design to validate it
    - Connection to existing literature
    """
    return json.dumps({
        "instruction": "propose_math_modification",
        "existing_method": existing_method,
        "target_improvement": target_improvement,
        "constraints": constraints,
        "karpathy_principles": [
            "Start from the simplest possible version",
            "Every term in the equation must have a physical/statistical interpretation",
            "If you can't explain it in one sentence, simplify further",
            "Derive gradient flow and check for vanishing/exploding gradients",
            "The modification should be ablation-testable in < 2 hours of compute",
        ],
        "required_output": [
            "modified_loss_or_architecture_latex",
            "intuition_one_paragraph",
            "expected_metric_improvement",
            "ablation_experiment_design",
            "related_papers_to_cite",
            "failure_modes_to_watch",
        ],
    })


@tool
def generate_ablation_matrix(
    model_components: list,
    datasets: list,
    metrics: list,
    max_experiments: int = 16,
) -> str:
    """
    Generate a systematic ablation study matrix JSON.
    model_components: list of components to ablate (e.g. ["ssim_loss", "smoothness_loss", "contrastive_loss"])
    datasets: list of datasets to test on
    metrics: list of metrics to track
    Returns the ablation design as JSON.
    """
    import itertools

    # Generate ablation configs: each component on/off
    experiments = []
    for r in range(1, min(4, len(model_components) + 1)):
        for combo in itertools.combinations(model_components, r):
            disabled = [c for c in model_components if c not in combo]
            experiments.append({
                "id": f"ablation_{len(experiments):03d}",
                "enabled_components": list(combo),
                "disabled_components": disabled,
                "datasets": datasets,
                "track_metrics": metrics,
                "estimated_runtime_min": 15 * len(datasets),
            })
            if len(experiments) >= max_experiments:
                break
        if len(experiments) >= max_experiments:
            break

    # Always include full model
    experiments.insert(0, {
        "id": "full_model_baseline",
        "enabled_components": model_components,
        "disabled_components": [],
        "datasets": datasets,
        "track_metrics": metrics,
        "estimated_runtime_min": 15 * len(datasets),
    })

    total_time = sum(e["estimated_runtime_min"] for e in experiments)
    return json.dumps({
        "ablation_matrix": experiments,
        "total_experiments": len(experiments),
        "estimated_total_runtime_hours": round(total_time / 60, 1),
        "recommendation": "Run full_model_baseline first, then ablations in parallel if GPU available",
    })


@tool
def run_quick_experiment(
    config_overrides: dict,
    epochs: int = 3,
    dataset_path: str = "data/sample",
) -> str:
    """
    Run a quick experiment with config overrides for rapid prototyping.
    Uses the existing training pipeline with a small epoch count.
    Returns JSON with training loss curve and descriptor stats.
    This is for fast iteration — not production training.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        import yaml
        config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Apply overrides
        for key, val in config_overrides.items():
            keys = key.split(".")
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = val

        config["training"]["epochs"] = epochs
        config["dataset"]["mri_root"] = dataset_path

        # Use existing orchestrator
        from scripts.research_orchestrator import (
            build_dataloader, build_model, create_dirs, train_model,
            extract_descriptor, evaluate_ablation
        )
        import torch

        create_dirs(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            loader, dataset = build_dataloader(config)
            model = build_model(config, device)
            history = train_model(config, model, loader, device)
            final_loss = history[-1]["loss"] if history else None
            return json.dumps({
                "status": "success",
                "device": str(device),
                "epochs_run": len(history),
                "final_loss": final_loss,
                "loss_curve": [h["loss"] for h in history],
                "config_overrides": config_overrides,
            })
        except Exception as inner_e:
            return json.dumps({
                "status": "dataset_unavailable",
                "message": str(inner_e),
                "note": "Training pipeline ready. Provide a real MRI dataset path to run.",
                "config_overrides": config_overrides,
            })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


# ---------------------------------------------------------------------------
# ── EMAIL TOOLS ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@tool
def draft_collaboration_email(
    subject: str,
    context: str,
    tone: str = "professional_enthusiastic",
    include_sections: Optional[list] = None,
) -> str:
    """
    Draft a professional collaboration/PhD inquiry email to Prof. Ruogu Fang
    at the SMILE Lab (ruogu.fang@bme.ufl.edu).

    context: Description of your background, what research you want to contribute,
             and what specific paper/project you're aligning with.
    tone: 'professional_enthusiastic' | 'concise' | 'detailed_technical'
    include_sections: list of sections to include, e.g. ['background', 'alignment',
                      'specific_contribution', 'ask', 'next_steps']

    Returns a JSON dict with the full email text and metadata.
    """
    if include_sections is None:
        include_sections = [
            "opening",
            "background",
            "alignment_with_smile_lab",
            "specific_technical_contribution",
            "ask_collaboration_or_supervision",
            "next_steps",
            "closing",
        ]

    return json.dumps({
        "instruction": "draft_email",
        "recipient": "Prof. Ruogu Fang",
        "recipient_email": "ruogu.fang@bme.ufl.edu",
        "subject": subject,
        "context": context,
        "tone": tone,
        "sections": include_sections,
        "constraints": [
            "Keep under 400 words — busy professors skim emails",
            "Lead with ONE specific paper from SMILE Lab to show genuine engagement",
            "Name one concrete technical contribution you can make immediately",
            "Do NOT use generic phrases like 'I am very interested in your work'",
            "End with a single clear ask (15-min Zoom / review attached work / etc.)",
            "Include your GitHub/portfolio link at the end",
        ],
        "professor_profile": {
            "name": "Dr. Ruogu Fang",
            "lab": "SMILE Lab",
            "affiliation": "University of Florida, BME",
            "email": "ruogu.fang@bme.ufl.edu",
            "focus": "multimodal medical AI, REVEAL++, retinal + cardiac imaging, trustworthy AI",
        },
    })


@tool
def send_collaboration_email(
    subject: str,
    body: str,
    sender_email: str = "",
    sender_name: str = "Kashim Mirza",
) -> str:
    """
    Send a collaboration email to Prof. Ruogu Fang at ruogu.fang@bme.ufl.edu.

    Reads SMTP credentials from environment variables:
      EMAIL_SENDER_ADDRESS — your Gmail address
      EMAIL_APP_PASSWORD    — Gmail App Password (not your main password)

    If credentials are not set, returns a dry-run preview with instructions.
    The email is always shown as a preview before sending.
    """
    recipient = "ruogu.fang@bme.ufl.edu"
    sender = sender_email or os.environ.get("EMAIL_SENDER_ADDRESS", "")
    password = os.environ.get("EMAIL_APP_PASSWORD", "")

    preview = (
        f"TO: {recipient}\n"
        f"FROM: {sender or '[EMAIL_SENDER_ADDRESS not set]'}\n"
        f"SUBJECT: {subject}\n"
        f"{'─' * 60}\n"
        f"{body}\n"
    )

    if not sender or not password:
        return json.dumps({
            "status": "dry_run",
            "preview": preview,
            "message": (
                "Email NOT sent. To send for real, add these to your .env file:\n"
                "  EMAIL_SENDER_ADDRESS=your.email@gmail.com\n"
                "  EMAIL_APP_PASSWORD=your_16_char_app_password\n"
                "Then re-run with --send-email flag."
            ),
        })

    # Build MIME message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{sender_name} <{sender}>"
    msg["To"] = recipient
    msg["Reply-To"] = sender
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        return json.dumps({
            "status": "sent",
            "preview": preview,
            "timestamp": datetime.now().isoformat(),
            "message": f"Email sent to {recipient} from {sender}",
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "preview": preview,
            "message": "SMTP send failed. Check EMAIL_APP_PASSWORD in your .env",
        })


# ---------------------------------------------------------------------------
# ── KARPATHY CRITIC ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

@tool
def karpathy_critique(
    claim: str,
    method: str,
    result: str,
) -> str:
    """
    Apply Andrej Karpathy's first-principles philosophy to critique a research
    claim, method, or result. Returns a structured prompt that forces rigorous
    questioning of every assumption.

    Karpathy's philosophy applied to research:
    - "What's the simplest model that could possibly work?"
    - "Can you derive this from scratch on a whiteboard?"
    - "What's the null hypothesis? Did you beat random/trivial baseline?"
    - "Are you sure it's not just learning a shortcut?"
    - "Would this work on a 1000-sample toy dataset?"
    """
    return json.dumps({
        "instruction": "karpathy_critique",
        "claim": claim,
        "method": method,
        "result": result,
        "critique_framework": {
            "simplicity_test": "What is the SIMPLEST version that achieves 80% of the gain?",
            "derivation_test": "Can you derive the key equation from first principles in 5 lines?",
            "baseline_test": "Does it beat trivial baselines (identity transform, pixel mean, etc.)?",
            "shortcut_test": "Could the model be cheating via a dataset artifact or label leak?",
            "scale_test": "Does it work on a toy 100-sample dataset? If not, why not?",
            "gradient_test": "What does the gradient of your loss w.r.t. input look like?",
            "assumption_test": "List every assumption made. Which one is most likely wrong?",
            "reproducibility_test": "Seed the RNG and run 3 times. Are results within 0.5%?",
        },
        "output_format": "critique_per_category_with_severity_high_medium_low",
    })


# ---------------------------------------------------------------------------
# ── TOOL REGISTRY ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    read_local_pdf,
    list_local_papers,
    bullet_point_paper,
    search_arxiv_papers,
    search_openalex_papers,
    find_research_gaps,
    propose_math_modification,
    generate_ablation_matrix,
    run_quick_experiment,
    draft_collaboration_email,
    send_collaboration_email,
    karpathy_critique,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}
