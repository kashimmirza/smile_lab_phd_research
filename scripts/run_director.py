#!/usr/bin/env python3
"""
scripts/run_director.py
------------------------
CLI entry point for the SMILE Lab Research Director Agent.

Usage examples:
  # Full pipeline (no experiments, no email send)
  python scripts/run_director.py

  # Full pipeline + run quick experiments
  python scripts/run_director.py --run-experiments

  # Full pipeline + send email to Prof. Fang
  python scripts/run_director.py --send-email

  # Full pipeline, run experiments AND send email
  python scripts/run_director.py --run-experiments --send-email

  # Custom research focus
  python scripts/run_director.py --focus "self-supervised OCTA vessel segmentation"

  # Use OpenAI instead of Gemini
  python scripts/run_director.py --llm openai

  # Save full output to JSON
  python scripts/run_director.py --output outputs/director_session.json

  # Skip to a specific node (for debugging)
  python scripts/run_director.py --start-from gap_finder
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║          SMILE Lab Research Director — LangGraph Agent           ║
║   Karpathy First-Principles × Prof. Fang's REVEAL++ Vision      ║
╚══════════════════════════════════════════════════════════════════╝

  Nodes: paper_reader → literature_scout → gap_finder →
         math_innovator → experiment_runner → results_analyst →
         latex_writer → email_drafter → karpathy_critic

"""
    print(banner)


def print_section(title: str, content: str, width: int = 70):
    """Pretty-print a section with a header."""
    print(f"\n{'━' * width}")
    print(f"  📄 {title}")
    print(f"{'━' * width}")
    if content:
        # Word-wrap long lines
        for line in content.split("\n"):
            if len(line) > width:
                for chunk in textwrap.wrap(line, width=width):
                    print(f"  {chunk}")
            else:
                print(f"  {line}" if line.strip() else "")
    else:
        print("  (no output)")


def save_session_output(state: dict, output_path: str):
    """Save the full session state to a JSON file (excluding the LLM client object)."""
    saveable = {
        k: v for k, v in state.items()
        if k != "llm_client" and isinstance(v, (str, int, float, bool, list, dict, type(None)))
    }
    saveable["_saved_at"] = datetime.now().isoformat()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(saveable, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Full session saved to: {output_path}")


def check_env():
    """Check environment and warn about missing keys."""
    print("🔑 Environment check:")
    keys = {
        "GEMINI_API_KEY": "Google Gemini (recommended)",
        "OPENAI_API_KEY": "OpenAI GPT-4",
        "ANTHROPIC_API_KEY": "Anthropic Claude",
        "EMAIL_SENDER_ADDRESS": "Email sending (optional)",
        "EMAIL_APP_PASSWORD": "Email sending (optional)",
    }
    has_llm = False
    for key, desc in keys.items():
        val = os.environ.get(key, "")
        status = "✅" if val else "❌"
        masked = val[:4] + "..." if val else "NOT SET"
        print(f"  {status} {key:<25} ({desc}) = {masked}")
        if key in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY") and val:
            has_llm = True
    if not has_llm:
        print("\n  ⚠️  No LLM API key found — stub LLM will be used.")
        print("     Add GEMINI_API_KEY=your_key to .env for real analysis.\n")
    return has_llm


def main():
    parser = argparse.ArgumentParser(
        description="SMILE Lab Research Director Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--focus",
        default="multimodal MRI and OCTA fusion for cardiac and retinal biomarker discovery aligned with SMILE Lab REVEAL++",
        help="Research focus for the agent (default: SMILE Lab multimodal focus)",
    )
    parser.add_argument(
        "--run-experiments",
        action="store_true",
        help="Run quick 3-epoch training experiments during the pipeline",
    )
    parser.add_argument(
        "--send-email",
        action="store_true",
        help="Send the drafted email to Prof. Fang via SMTP (requires EMAIL_* in .env)",
    )
    parser.add_argument(
        "--llm",
        choices=["google", "openai", "anthropic"],
        default="google",
        help="LLM provider to use (default: google/Gemini)",
    )
    parser.add_argument(
        "--output",
        default="outputs/director_session.json",
        help="Path to save the full session JSON output",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (wait for full completion)",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Only check environment variables and exit",
    )

    args = parser.parse_args()

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    print_banner()
    has_llm = check_env()

    if args.check_env:
        return

    print(f"\n🎯 Research Focus: {args.focus}")
    print(f"🧪 Run Experiments: {args.run_experiments}")
    print(f"📧 Send Email: {args.send_email}")
    print(f"🤖 LLM Provider: {args.llm}")
    print(f"💾 Output: {args.output}")
    print()

    # Run the director
    from agents.langgraph_director import run_director

    try:
        final_state = run_director(
            research_focus=args.focus,
            run_experiments=args.run_experiments,
            send_email=args.send_email,
            llm_provider=args.llm,
            stream=not args.no_stream,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Partial results may be available.")
        return
    except Exception as e:
        print(f"\n❌ Error running director: {e}")
        import traceback
        traceback.print_exc()
        return

    # ── Display results ────────────────────────────────────────────────────
    print("\n\n" + "═" * 70)
    print("  🎓 RESEARCH DIRECTOR COMPLETE — Summary")
    print("═" * 70)

    # Paper summaries
    papers = final_state.get("paper_summaries", [])
    print(f"\n📚 Papers analyzed: {len(papers)}")
    for p in papers:
        print(f"   • {p.get('file', 'unknown')}")

    # Literature
    sota = final_state.get("sota_papers", {})
    print(f"📰 SOTA papers found: {len(sota.get('arxiv', []))} ArXiv + {len(sota.get('openalex', []))} OpenAlex")

    print_section("Literature Synthesis", final_state.get("literature_synthesis", ""))
    print_section("Research Gaps", final_state.get("research_gaps", ""))
    print_section("Mathematical Innovations", final_state.get("mathematical_innovations", ""))
    print_section("Experiment Plan", final_state.get("experiment_plan", ""))
    print_section("Results Analysis", final_state.get("results_analysis", ""))
    print_section("LaTeX Sections (Preview)", (final_state.get("latex_sections", "") or "")[:1500] + "...")

    # Email
    email = final_state.get("email_draft", {})
    if email:
        print(f"\n{'━' * 70}")
        print("  📧 EMAIL DRAFT")
        print(f"{'━' * 70}")
        print(f"  TO: {email.get('to', '')}")
        print(f"  SUBJECT: {email.get('subject', '')}")
        print(f"{'─' * 70}")
        print(email.get("body", ""))
        send_result = email.get("send_result")
        if send_result:
            status = send_result.get("status", "unknown")
            print(f"\n  📤 Send status: {status.upper()}")
            if status == "dry_run":
                print("  💡 Add EMAIL_SENDER_ADDRESS + EMAIL_APP_PASSWORD to .env to send for real")

    # Karpathy critique
    print_section("Karpathy Critique", final_state.get("karpathy_critique_output", ""))

    # Director log
    log = final_state.get("director_log", [])
    if log:
        print(f"\n📋 Director Log ({len(log)} entries):")
        for entry in log:
            print(f"   {entry}")

    # Save full output
    save_session_output(final_state, args.output)

    print(f"\n🏁 Done. Full session at: {args.output}")


if __name__ == "__main__":
    main()
