"""
agents/research_agent.py
--------------------------
A ReAct-style agent: at each step, Claude either calls one tool from
agents/tools.py or produces a final answer. Every step is logged to
outputs/agent_log.json so the run is reproducible and auditable --
this is meant to survive a PhD committee asking "what exactly did the
agent do," not just "did it work."
"""

import json
import os
from datetime import datetime

import anthropic

from agents.session_state import ResearchSession
from agents.tools import TOOL_SCHEMAS, dispatch

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are a biomedical research assistant agent working inside the SMILE Lab
PhD research repo. You orchestrate an existing, already-validated pipeline for 3D MRI/OCTA
motion-descriptor registration research -- you do not invent new science or bypass
data-access rules.

Rules you must always follow:
- Never attempt to circumvent a dataset's access gate. If resolve_datasets reports
  'manual_action_required' for a dataset (e.g. ADNI, UK Biobank), report that plainly to the
  user as a next step for them to complete manually -- do not try alternate sources to get
  around a data use agreement.
- Follow the pipeline's real dependency order: build_model_and_data -> train_model ->
  extract_motion_descriptor -> run_ablation -> generate_report. Don't skip a stage that a
  later tool depends on.
- After each tool result, briefly state what you learned and what you'll do next, before
  calling the next tool.
- Stop and report back to the user (rather than continuing) if a tool result indicates a
  human decision is required -- e.g., a dataset needs manual access approval, or training
  loss is not decreasing.
- When the goal is achieved (or cannot proceed further without human input), give a concise
  final summary: what was resolved, what was trained, key metrics, and where outputs live.
"""


class ResearchAgent:
    def __init__(self, config: dict, max_steps: int = 15):
        self.session = ResearchSession(config=config)
        self.max_steps = max_steps
        self.client = anthropic.Anthropic()  # expects ANTHROPIC_API_KEY in environment

    def run(self, goal: str) -> str:
        messages = [{"role": "user", "content": goal}]

        for step in range(self.max_steps):
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            tool_calls = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b.text for b in response.content if b.type == "text"]
            for t in text_blocks:
                print(f"\n[step {step}] {t}")

            if not tool_calls:
                # Agent produced a final answer, no more tool calls.
                self._save_log()
                return "\n".join(text_blocks)

            tool_results = []
            for call in tool_calls:
                print(f"[step {step}] -> calling {call.name}({call.input})")
                result_text = dispatch(call.name, call.input, self.session)
                print(f"[step {step}] <- {result_text}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": result_text,
                })

            messages.append({"role": "user", "content": tool_results})

        self._save_log()
        return "Reached max_steps without a final answer. Check outputs/agent_log.json for progress."

    def _save_log(self):
        output_dir = self.session.config.get("paths", {}).get("output_dir", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "agent_log.json")
        with open(path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "actions": self.session.action_log,
            }, f, indent=2)
        print(f"\nAgent action log saved to {path}")