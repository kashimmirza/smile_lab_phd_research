"""
agents/langgraph_state.py
--------------------------
Typed state definition for the SMILE Lab Research Director LangGraph.
Uses TypedDict so all nodes have validated, documented state keys.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ResearchDirectorState(TypedDict, total=False):
    """
    Shared state across all nodes in the Research Director graph.
    All fields are optional (total=False) — nodes only set what they produce.
    """

    # ── Configuration ───────────────────────────────────────────────────────
    llm_client: Any                    # Initialized LLM client (Google GenAI, OpenAI, etc.)
    model_name: str                    # e.g. "gemini-2.0-flash", "gpt-4o"
    research_focus: str                # e.g. "multimodal MRI+OCTA registration"
    run_experiments: bool              # Whether to actually run training
    send_email: bool                   # Whether to actually send the email via SMTP

    # ── Paper Reading ────────────────────────────────────────────────────────
    paper_summaries: List[Dict]        # List of {file, analysis, word_count, ...}

    # ── Literature ───────────────────────────────────────────────────────────
    sota_papers: Dict                  # {arxiv: [...], openalex: [...]}
    literature_synthesis: str          # LLM synthesis of SOTA landscape

    # ── Gap Analysis ─────────────────────────────────────────────────────────
    research_gaps: str                 # Ranked list of research gaps

    # ── Math Innovation ──────────────────────────────────────────────────────
    mathematical_innovations: str      # Proposed equations and modifications

    # ── Experiments ──────────────────────────────────────────────────────────
    ablation_design: Dict              # Ablation matrix JSON
    experiment_result: Optional[Dict]  # Quick experiment result (if run)
    experiment_plan: str               # Prioritized experiment plan

    # ── Results Analysis ─────────────────────────────────────────────────────
    results_analysis: str              # Statistical analysis + figure descriptions

    # ── LaTeX Writing ────────────────────────────────────────────────────────
    latex_sections: str                # Drafted LaTeX (Introduction, Methods, Experiments)

    # ── Email ────────────────────────────────────────────────────────────────
    email_draft: Dict                  # {subject, to, body, send_result}

    # ── Critic ───────────────────────────────────────────────────────────────
    karpathy_critique_output: str      # Final critique and publishability score

    # ── Meta ─────────────────────────────────────────────────────────────────
    director_log: List[str]            # Audit log of all node actions
    current_node: str                  # Which node is currently active
    error: Optional[str]               # Error message if something failed
