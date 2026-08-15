"""
agents/langgraph_director.py
------------------------------
SMILE Lab Research Director — the main LangGraph state graph.

This is the top-level orchestrator. It wires together all 9 specialized
nodes into a directed graph with conditional routing.

Graph flow:
  START
    → paper_reader       (read local PDFs)
    → literature_scout   (search SOTA)
    → gap_finder         (identify gaps)
    → math_innovator     (propose modifications)
    → experiment_runner  (design ablation)
    → results_analyst    (interpret results)
    → latex_writer       (draft paper)
    → email_drafter      (write + optionally send email)
    → karpathy_critic    (final critique)
    → END

Each node is a pure function (state) -> dict patch.
The supervisor node decides whether to continue or stop early.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from agents.langgraph_state import ResearchDirectorState
from agents.langgraph_nodes import (
    email_drafter_node,
    experiment_runner_node,
    gap_finder_node,
    karpathy_critic_node,
    latex_writer_node,
    literature_scout_node,
    math_innovator_node,
    paper_reader_node,
    results_analyst_node,
)


# ---------------------------------------------------------------------------
# Supervisor / router
# ---------------------------------------------------------------------------

def supervisor_router(state: ResearchDirectorState) -> Literal[
    "paper_reader",
    "literature_scout",
    "gap_finder",
    "math_innovator",
    "experiment_runner",
    "results_analyst",
    "latex_writer",
    "email_drafter",
    "karpathy_critic",
    "__end__",
]:
    """
    Conditional router that decides which node to visit next.
    Called after each node completes.

    The routing is sequential by default, but can skip nodes if
    the state already has the required data (e.g., for resume runs).
    """
    current = state.get("current_node", "")

    # Fatal error: stop
    if state.get("error"):
        print(f"[SUPERVISOR] Error detected: {state['error']}. Stopping.")
        return "__end__"

    # Sequential routing
    sequence = [
        "paper_reader",
        "literature_scout",
        "gap_finder",
        "math_innovator",
        "experiment_runner",
        "results_analyst",
        "latex_writer",
        "email_drafter",
        "karpathy_critic",
    ]

    if current == "":
        return "paper_reader"

    try:
        idx = sequence.index(current)
        if idx + 1 < len(sequence):
            next_node = sequence[idx + 1]
            print(f"[SUPERVISOR] {current} → {next_node}")
            return next_node
        else:
            print("[SUPERVISOR] All nodes complete → END")
            return "__end__"
    except ValueError:
        return "__end__"


# ---------------------------------------------------------------------------
# Node wrappers that update current_node
# ---------------------------------------------------------------------------

def _wrap(node_fn, node_name: str):
    """Wrap a node function to track current_node in state."""
    def wrapped(state: ResearchDirectorState) -> dict:
        print(f"\n{'='*60}")
        print(f"  🔬 NODE: {node_name.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        patch = node_fn(state)
        patch["current_node"] = node_name
        return patch
    wrapped.__name__ = node_name
    return wrapped


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_director_graph(checkpointing: bool = True) -> StateGraph:
    """
    Build and compile the SMILE Lab Research Director LangGraph.

    Args:
        checkpointing: If True, uses MemorySaver for node-level checkpointing.
                       This allows resuming a run from any node.

    Returns:
        Compiled LangGraph StateGraph ready for .invoke() or .stream()
    """
    builder = StateGraph(ResearchDirectorState)

    # Register all nodes (wrapped for tracking)
    builder.add_node("paper_reader",      _wrap(paper_reader_node,      "paper_reader"))
    builder.add_node("literature_scout",  _wrap(literature_scout_node,  "literature_scout"))
    builder.add_node("gap_finder",        _wrap(gap_finder_node,        "gap_finder"))
    builder.add_node("math_innovator",    _wrap(math_innovator_node,    "math_innovator"))
    builder.add_node("experiment_runner", _wrap(experiment_runner_node, "experiment_runner"))
    builder.add_node("results_analyst",   _wrap(results_analyst_node,   "results_analyst"))
    builder.add_node("latex_writer",      _wrap(latex_writer_node,      "latex_writer"))
    builder.add_node("email_drafter",     _wrap(email_drafter_node,     "email_drafter"))
    builder.add_node("karpathy_critic",   _wrap(karpathy_critic_node,   "karpathy_critic"))

    # Entry point
    builder.add_edge(START, "paper_reader")

    # Sequential edges (each node goes to the next)
    builder.add_edge("paper_reader",      "literature_scout")
    builder.add_edge("literature_scout",  "gap_finder")
    builder.add_edge("gap_finder",        "math_innovator")
    builder.add_edge("math_innovator",    "experiment_runner")
    builder.add_edge("experiment_runner", "results_analyst")
    builder.add_edge("results_analyst",   "latex_writer")
    builder.add_edge("latex_writer",      "email_drafter")
    builder.add_edge("email_drafter",     "karpathy_critic")
    builder.add_edge("karpathy_critic",   END)

    # Compile
    kwargs = {}
    if checkpointing:
        kwargs["checkpointer"] = MemorySaver()

    return builder.compile(**kwargs)


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def _init_llm_client(provider: str = "google"):
    """Initialize the LLM client based on available API keys."""
    if provider == "google":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                return genai.GenerativeModel("gemini-2.0-flash")
            except ImportError:
                pass

    if provider in ("openai", "google"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
            except ImportError:
                pass

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-3-5-haiku-20241022", api_key=api_key)
        except ImportError:
            pass

    # Fallback: stub client for testing
    class StubLLM:
        def generate_content(self, prompt: str):
            class R:
                text = f"[STUB LLM] No API key found. Set GEMINI_API_KEY. Prompt: {prompt[:100]}"
            return R()
        def invoke(self, messages):
            class R:
                content = f"[STUB LLM] No API key configured."
            return R()
    print("⚠️  No LLM API key found. Using stub. Set GEMINI_API_KEY in .env")
    return StubLLM()


def run_director(
    research_focus: str = "multimodal MRI and OCTA fusion for cardiac and retinal biomarker discovery",
    run_experiments: bool = False,
    send_email: bool = False,
    llm_provider: str = "google",
    stream: bool = True,
) -> ResearchDirectorState:
    """
    Run the full SMILE Lab Research Director pipeline.

    Args:
        research_focus: The high-level research question to investigate.
        run_experiments: If True, actually run 3-epoch training experiments.
        send_email: If True, send the drafted email via SMTP (requires .env).
        llm_provider: "google" | "openai" | "anthropic"
        stream: If True, stream node outputs as they complete.

    Returns:
        Final ResearchDirectorState with all outputs.
    """
    from dotenv import load_dotenv
    load_dotenv()

    llm_client = _init_llm_client(llm_provider)

    initial_state: ResearchDirectorState = {
        "llm_client": llm_client,
        "model_name": "gemini-2.0-flash",
        "research_focus": research_focus,
        "run_experiments": run_experiments,
        "send_email": send_email,
        "director_log": [],
        "current_node": "",
    }

    graph = build_director_graph(checkpointing=True)
    config = {"configurable": {"thread_id": "smile_lab_session_1"}}

    if stream:
        print("\n🔬 SMILE Lab Research Director — Starting\n")
        final_state = initial_state.copy()
        for event in graph.stream(initial_state, config=config):
            for node_name, state_patch in event.items():
                if node_name == "__end__":
                    continue
                final_state.update(state_patch)
                print(f"\n✅ Node '{node_name}' complete.")
        return final_state
    else:
        return graph.invoke(initial_state, config=config)
