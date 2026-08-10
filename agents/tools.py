"""
agents/tools.py
-----------------
Wraps your EXISTING functions (dataset_crawler.py, research_orchestrator.py)
as tools the agent can call. No new science logic here on purpose -- the
agent should orchestrate what you already built and validated, not
reimplement it.

Each tool has:
  - a JSON schema (for Claude's tool-use API)
  - a dispatch function that takes (tool_input, session) -> str summary

Gated datasets (ADNI, UK Biobank, etc.) are never auto-bypassed -- the
underlying fetch_dataset() call already returns "manual_action_required"
for these, and the agent only ever sees that status, never a workaround.
"""

import os
from data.dataset_resolver import resolve_dataset_root
from scripts.dataset_crawler import (
    extract_dataset_mentions,
    fetch_by_title,
    fetch_dataset,
    write_manifest,
)
from scripts.research_orchestrator import (
    build_dataloader,
    build_model,
    create_dirs,
    evaluate_ablation,
    extract_descriptor,
    generate_latex_report,
    train_model,
)
from agents.latex_writer import build_full_paper, write_latex_section
from agents.social_writer import write_all_platforms, write_social_post
from agents.equation_animator import animate_directional_cosine, animate_generic_equation

import json
import torch

TOOL_SCHEMAS = [
    {
        "name": "search_paper",
        "description": (
            "Look up a paper by title (via Semantic Scholar) and return its title/abstract. "
            "Use this first when the goal references a specific paper or research question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"title": {"type": "string", "description": "Paper title or close paraphrase"}},
            "required": ["title"],
        },
    },
    {
        "name": "resolve_datasets",
        "description": (
            "Detect datasets referenced in the last searched paper (or a given title), and attempt "
            "to fetch each from its open API. Gated datasets (ADNI, UK Biobank, CheXpert, etc.) are "
            "never auto-downloaded -- they come back as 'manual_action_required' with instructions. "
            "Updates dataset.mri_root / dataset.octa_root in the session config if a matching "
            "mri_dataset_name / octa_dataset_name is set."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_title": {
                    "type": "string",
                    "description": "Optional. If omitted, uses the most recently searched paper.",
                }
            },
        },
    },
    {
        "name": "build_model_and_data",
        "description": "Instantiate the registration model and MRI dataloader from the current config. Must run before train_model.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "train_model",
        "description": (
            "Train the registration model on the currently loaded data. Optionally override the "
            "epoch count for a quick agent iteration instead of a full run."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "epochs_override": {
                    "type": "integer",
                    "description": "If set, trains for this many epochs instead of config.training.epochs.",
                }
            },
        },
    },
    {
        "name": "extract_motion_descriptor",
        "description": "Run the trained model on one sample and extract its motion descriptor. Requires train_model to have run first.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_ablation",
        "description": "Run the ablation study matrix and save metrics.json. Requires extract_motion_descriptor to have run first.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_report",
        "description": "Generate the basic auto-formatted LaTeX metrics report from the current metrics. Requires run_ablation to have run first.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "write_latex_section",
        "description": (
            "Draft a single LaTeX section (e.g. Methods, a specific equation, a paragraph "
            "explaining the loss function) on instruction. Equations are grounded in the "
            "pipeline's actual implemented math, and cited metrics use this session's real "
            "results if available. Returns the LaTeX text directly (does not write a file)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "What to write, e.g. 'Write the Methods subsection deriving the total loss function' or 'Write the equation for the directional cosine descriptor with a one-sentence explanation'.",
                }
            },
            "required": ["instruction"],
        },
    },
    {
        "name": "write_journal_paper",
        "description": (
            "Assemble a full journal-style LaTeX paper across multiple sections, each drafted "
            "on its own instruction and grounded in this session's real training/ablation "
            "results. Writes a .tex file, and compiles it to PDF if a LaTeX engine is available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "author": {"type": "string"},
                "sections": {
                    "type": "array",
                    "description": "Ordered list of sections to write.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "e.g. 'Abstract', 'Methods', 'Results'"},
                            "instruction": {"type": "string", "description": "What this section should cover"},
                        },
                        "required": ["name", "instruction"],
                    },
                },
                "output_path": {
                    "type": "string",
                    "description": "Where to write the .tex file, e.g. 'outputs/paper.tex'",
                },
                "compile_pdf": {
                    "type": "boolean",
                    "description": "If true, attempt to compile the .tex to PDF (requires pdflatex or tectonic on PATH).",
                },
            },
            "required": ["title", "author", "sections", "output_path"],
        },
    },
    {
        "name": "write_social_post",
        "description": (
            "Draft one social media post for a single platform (linkedin, twitter, facebook, "
            "or instagram), respecting that platform's practical length/style conventions. "
            "Grounded in this session's real results -- never fabricates numbers. Equations are "
            "described in plain words, not LaTeX, since these platforms don't render LaTeX."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "platform": {"type": "string", "enum": ["linkedin", "twitter", "facebook", "instagram"]},
                "instruction": {"type": "string", "description": "What the post should focus on / announce"},
            },
            "required": ["platform", "instruction"],
        },
    },
    {
        "name": "write_social_campaign",
        "description": "Draft posts for multiple platforms at once from the same instruction, each tailored to its own length/style.",
        "input_schema": {
            "type": "object",
            "properties": {
                "instruction": {"type": "string"},
                "platforms": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["linkedin", "twitter", "facebook", "instagram"]},
                    "description": "Omit to generate for all four platforms.",
                },
            },
            "required": ["instruction"],
        },
    },
    {
        "name": "animate_equation",
        "description": (
            "Generate an animated GIF visualizing a mathematical equation, for attaching to a "
            "social post or for teaching/understanding. mode='directional_cosine' renders the "
            "pre-built, codebase-grounded animation of the motion descriptor (phi rotating "
            "against reference r, with alpha=cos(theta) traced live). mode='generic' animates "
            "any sympy-parseable f(x, t) as t sweeps a range -- use this for other equations "
            "(e.g. 'sin(x - t)' for a traveling wave, or a custom loss-landscape slice)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["directional_cosine", "generic"]},
                "output_path": {"type": "string", "description": "e.g. 'outputs/cosine_animation.gif'"},
                "expr": {"type": "string", "description": "Required if mode='generic'. Sympy expression in terms of var and t_var, e.g. 'sin(x - t)'"},
                "var": {"type": "string", "description": "Spatial variable name, default 'x'"},
                "t_var": {"type": "string", "description": "Animated parameter name, default 't'"},
                "t_start": {"type": "number"},
                "t_end": {"type": "number"},
                "n_frames": {"type": "integer"},
                "x_start": {"type": "number"},
                "x_end": {"type": "number"},
                "n_points": {"type": "integer"},
                "title": {"type": "string"},
            },
            "required": ["mode", "output_path"],
        },
    },
]


def dispatch(tool_name: str, tool_input: dict, session) -> str:
    if tool_name == "search_paper":
        ctx = fetch_by_title(tool_input["title"])
        session.last_paper_context = ctx
        summary = f"Found: '{ctx.title}'. Abstract (first 300 chars): {ctx.abstract[:300]}"
        session.log(tool_name, tool_input, summary)
        return summary

    if tool_name == "resolve_datasets":
        title = tool_input.get("paper_title")
        ctx = session.last_paper_context
        if title:
            ctx = fetch_by_title(title)
            session.last_paper_context = ctx
        if ctx is None:
            msg = "No paper context available. Call search_paper first, or pass paper_title."
            session.log(tool_name, tool_input, msg)
            return msg

        matches = extract_dataset_mentions(ctx)
        if not matches:
            msg = "No known datasets detected in this paper's text."
            session.log(tool_name, tool_input, msg)
            return msg

        results = [fetch_dataset(m) for m in matches]
        write_manifest(ctx.title, results)

        res_cfg = session.config.get("dataset_resolution", {})
        manifest_path = res_cfg.get("manifest_path", "data/dataset_manifest.json")
        mri_name = res_cfg.get("mri_dataset_name")
        if mri_name:
            session.config["dataset"]["mri_root"] = resolve_dataset_root(
                mri_name, manifest_path=manifest_path, fallback=session.config["dataset"]["mri_root"]
            )
        octa_name = res_cfg.get("octa_dataset_name")
        if octa_name:
            session.config["dataset"]["octa_root"] = resolve_dataset_root(
                octa_name, manifest_path=manifest_path, fallback=session.config["dataset"]["octa_root"]
            )

        summary = "; ".join(f"{r['name']}: {r['status']}" for r in results)
        session.log(tool_name, tool_input, summary)
        return summary

    if tool_name == "build_model_and_data":
        create_dirs(session.config)
        session.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        session.loader, session.dataset = build_dataloader(session.config)
        session.model = build_model(session.config, session.device)
        summary = f"Built model on {session.device}, {len(session.dataset)} samples loaded from {session.config['dataset']['mri_root']}"
        session.log(tool_name, tool_input, summary)
        return summary

    if tool_name == "train_model":
        if session.model is None or session.loader is None:
            msg = "Model/data not built yet. Call build_model_and_data first."
            session.log(tool_name, tool_input, msg)
            return msg
        cfg = session.config
        if "epochs_override" in tool_input:
            cfg = dict(cfg)
            cfg["training"] = dict(cfg["training"])
            cfg["training"]["epochs"] = tool_input["epochs_override"]
        session.history = train_model(cfg, session.model, session.loader, session.device)
        final_loss = session.history[-1]["loss"] if session.history else None
        summary = f"Trained {len(session.history)} epoch(s). Final loss: {final_loss}"
        session.log(tool_name, tool_input, summary)
        return summary

    if tool_name == "extract_motion_descriptor":
        if not session.history:
            msg = "No trained model yet. Call train_model first."
            session.log(tool_name, tool_input, msg)
            return msg
        session.descriptor_stats = extract_descriptor(
            session.model, session.dataset[0], session.config, session.config["paths"]["output_dir"]
        )
        summary = f"Descriptor extracted. Mean alpha: {session.descriptor_stats['mean_alpha']:.6f}"
        session.log(tool_name, tool_input, summary)
        return summary

    if tool_name == "run_ablation":
        if session.descriptor_stats is None:
            msg = "No descriptor yet. Call extract_motion_descriptor first."
            session.log(tool_name, tool_input, msg)
            return msg
        session.metrics = evaluate_ablation(
            session.history, session.descriptor_stats, session.config, session.config["paths"]["output_dir"]
        )
        summary = f"Ablation complete. {len(session.metrics['ablation_results'])} experiment(s) recorded."
        session.log(tool_name, tool_input, summary)
        return summary

    if tool_name == "generate_report":
        if session.metrics is None:
            msg = "No metrics yet. Call run_ablation first."
            session.log(tool_name, tool_input, msg)
            return msg
        session.report_path = generate_latex_report(session.metrics, session.config["paths"]["output_dir"])
        summary = f"Report written to {session.report_path}"
        session.log(tool_name, tool_input, summary)
        return summary

    if tool_name == "write_latex_section":
        tex = write_latex_section(tool_input["instruction"], session)
        summary = f"Drafted section ({len(tex)} chars). Preview: {tex[:150]}..."
        session.log(tool_name, tool_input, summary)
        # Return the full LaTeX so the agent can present or save it, not just a preview.
        return tex

    if tool_name == "write_journal_paper":
        sections = [(s["name"], s["instruction"]) for s in tool_input["sections"]]
        result = build_full_paper(
            session,
            title=tool_input["title"],
            author=tool_input["author"],
            section_instructions=sections,
            output_path=tool_input["output_path"],
            compile_pdf=tool_input.get("compile_pdf", False),
        )
        if result["compiled"]:
            summary = f"Paper written and compiled: {result['tex_path']} -> {result['pdf_path']}"
        else:
            note = result.get("compile_note", "")
            summary = f"Paper written to {result['tex_path']} (not compiled to PDF). {note}"
        session.log(tool_name, tool_input, summary)
        return summary

    if tool_name == "write_social_post":
        text = write_social_post(tool_input["platform"], tool_input["instruction"], session)
        summary = f"[{tool_input['platform']}] {len(text)} chars drafted."
        session.log(tool_name, tool_input, summary)
        return text

    if tool_name == "write_social_campaign":
        posts = write_all_platforms(tool_input["instruction"], session, tool_input.get("platforms"))
        summary = "; ".join(f"{p}: {len(t)} chars" for p, t in posts.items())
        session.log(tool_name, tool_input, summary)
        return json.dumps(posts, indent=2)

    if tool_name == "animate_equation":
        if tool_input["mode"] == "directional_cosine":
            path = animate_directional_cosine(tool_input["output_path"])
        else:
            path = animate_generic_equation(
                tool_input["expr"],
                tool_input.get("var", "x"),
                tool_input.get("t_var", "t"),
                (tool_input.get("t_start", 0), tool_input.get("t_end", 6.28318), tool_input.get("n_frames", 60)),
                (tool_input.get("x_start", -10), tool_input.get("x_end", 10), tool_input.get("n_points", 200)),
                tool_input["output_path"],
                title=tool_input.get("title", ""),
            )
        summary = f"Animation saved to {path}"
        session.log(tool_name, tool_input, summary)
        return summary

    return f"Unknown tool: {tool_name}"