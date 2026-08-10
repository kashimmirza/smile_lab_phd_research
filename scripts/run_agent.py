"""
scripts/run_agent.py
----------------------
Usage:
    python scripts/run_agent.py \
        --config configs/default_config.yaml \
        --goal "Find the paper on OCTA motion descriptors, resolve its dataset, \
                 train for 5 epochs, run the ablation study, and generate the report."

Requires ANTHROPIC_API_KEY set in your environment (see .env.example).
"""

import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import yaml

from agents.research_agent import ResearchAgent


def main():
    parser = argparse.ArgumentParser(description="Run the agentic research pipeline against a goal.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--goal", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=15)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    agent = ResearchAgent(config=config, max_steps=args.max_steps)
    final_answer = agent.run(args.goal)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(final_answer)


if __name__ == "__main__":
    main()