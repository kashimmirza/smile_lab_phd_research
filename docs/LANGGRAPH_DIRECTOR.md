<!-- @format -->

# SMILE Lab Research Director — LangGraph Agent

## What this does

A fully autonomous **LangGraph multi-agent system** that acts as your personal SMILE Lab research director. It reads your local papers, scans the SOTA literature, proposes novel ideas, designs experiments, drafts paper sections, and writes collaboration emails to Prof. Ruogu Fang — all in one command.

**No touch required** after `python scripts/run_director.py`.

---

## Node Pipeline

```
paper_reader → literature_scout → gap_finder → math_innovator
    → experiment_runner → results_analyst → latex_writer
    → email_drafter → karpathy_critic
```

| Node | What it does |
|---|---|
| `paper_reader` | Extracts and analyzes your local PDFs in `research_papers/` |
| `literature_scout` | Searches ArXiv + OpenAlex for 2023-2026 SOTA papers |
| `gap_finder` | Identifies novel research gaps vs. SOTA |
| `math_innovator` | Proposes new equations/modifications (Karpathy first-principles) |
| `experiment_runner` | Designs ablation matrix; optionally runs quick training |
| `results_analyst` | Interprets results, plans figures, statistical tests |
| `latex_writer` | Drafts Introduction, Methods, Experiments in LaTeX |
| `email_drafter` | Writes collaboration email to `ruogu.fang@bme.ufl.edu` |
| `karpathy_critic` | Challenges every claim — publishability score + action items |

---

## Quick Start

```bash
# 1. Install new dependencies
pip install langgraph langchain langchain-core langchain-google-genai pymupdf requests

# 2. Set up .env
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Run the full pipeline
python scripts/run_director.py

# 4. Run with experiments + email send
python scripts/run_director.py --run-experiments --send-email

# 5. Check environment only
python scripts/run_director.py --check-env
```

---

## CLI Options

```
python scripts/run_director.py [options]

Options:
  --focus TEXT         Custom research focus (default: SMILE Lab multimodal)
  --run-experiments    Run 3-epoch training experiments during pipeline
  --send-email         Send email to Prof. Fang via SMTP (needs .env)
  --llm PROVIDER       LLM: google (default) | openai | anthropic
  --output PATH        Save full session JSON (default: outputs/director_session.json)
  --no-stream          Disable streaming output
  --check-env          Check environment variables and exit
```

---

## Email Sending Setup

The agent **always drafts** the email. To **actually send** it:

1. Generate a [Gmail App Password](https://myaccount.google.com/apppasswords)
2. Add to `.env`:
   ```
   EMAIL_SENDER_ADDRESS=your.email@gmail.com
   EMAIL_APP_PASSWORD=xxxx_xxxx_xxxx_xxxx
   ```
3. Run with `--send-email` flag

---

## Output Files

| File | Contents |
|---|---|
| `outputs/director_session.json` | Full session state (all node outputs) |
| `outputs/latex/` | Generated LaTeX sections |
| `outputs/emails/` | Drafted emails |
| `outputs/agent_log.json` | Existing pipeline audit log |

---

## Persona: Andrej Karpathy × Prof. Fang

The `karpathy_critic` node applies these principles to every claim:

- **Simplicity test**: What's the simplest model that achieves 80% of the gain?
- **Derivation test**: Can you derive the key equation from first principles?
- **Baseline test**: Does it beat trivial baselines?
- **Shortcut test**: Could the model be cheating via a dataset artifact?
- **Scale test**: Does it work on a 100-sample toy dataset?

The `supervisor` node maintains Prof. Fang's SMILE Lab direction:
- Clinical translatability
- REVEAL++ alignment
- Multimodal rigor (MRI + OCTA + text)

---

## Architecture Note (LangChain Tool Calling)

All tools follow the LangChain `AssembledToolCall` pattern from the docs:
```python
@tool
def send_collaboration_email(subject: str, body: str) -> str:
    """Sends to ruogu.fang@bme.ufl.edu via SMTP"""
    ...
```

Each tool has: `name`, `args` (input), `output`, and implicit lifecycle status
(`running` → `finished` or `error`) managed by LangGraph's state machine.
