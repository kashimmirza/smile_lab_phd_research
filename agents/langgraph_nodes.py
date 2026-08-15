"""
agents/langgraph_nodes.py
--------------------------
Nine specialized node functions for the SMILE Lab Research Director.

Each node receives the shared ResearchDirectorState, calls tools or an LLM,
and returns a state patch. Nodes are pure functions: (state) -> dict.

Node roster:
  1. paper_reader_node       — reads local PDFs, extracts structure
  2. literature_scout_node   — searches ArXiv + OpenAlex for SOTA
  3. gap_finder_node         — identifies research gaps vs. SOTA
  4. math_innovator_node     — proposes mathematical modifications
  5. experiment_runner_node  — designs and runs ablation
  6. results_analyst_node    — interprets metrics, makes figures
  7. latex_writer_node       — drafts paper sections
  8. email_drafter_node      — writes collaboration email
  9. karpathy_critic_node    — challenges every claim (runs last)
"""

from __future__ import annotations

import json
from typing import Any

from agents.langgraph_state import ResearchDirectorState
from agents.langgraph_tools import (
    bullet_point_paper,
    draft_collaboration_email,
    find_research_gaps,
    generate_ablation_matrix,
    karpathy_critique,
    list_local_papers,
    propose_math_modification,
    read_local_pdf,
    run_quick_experiment,
    search_arxiv_papers,
    search_openalex_papers,
    send_collaboration_email,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm(state: ResearchDirectorState, prompt: str, system: str = "") -> str:
    """Call the LLM configured in state. Returns text response."""
    client = state["llm_client"]
    model = state.get("model_name", "gemini-2.0-flash")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Support both Google GenAI and OpenAI-compatible clients
    if hasattr(client, "generate_content"):
        # Google GenAI
        full_prompt = (system + "\n\n" + prompt) if system else prompt
        response = client.generate_content(full_prompt)
        return response.text
    else:
        # OpenAI-compatible (LangChain ChatOpenAI, Anthropic, etc.)
        resp = client.invoke(messages)
        return resp.content if hasattr(resp, "content") else str(resp)


def _log(state: ResearchDirectorState, node: str, msg: str):
    """Append to the director log."""
    entry = f"[{node.upper()}] {msg}"
    print(entry)
    return entry


# ---------------------------------------------------------------------------
# NODE 1: Paper Reader
# ---------------------------------------------------------------------------

def paper_reader_node(state: ResearchDirectorState) -> dict:
    """
    Read all local PDFs, extract text, produce bullet-point summaries.
    Stores results in state['paper_summaries'].
    """
    _log(state, "paper_reader", "Listing local papers...")

    raw = list_local_papers.invoke({"papers_dir": "research_papers"})
    paper_list = json.loads(raw)
    papers = paper_list.get("papers", [])

    if not papers:
        return {
            "paper_summaries": [],
            "director_log": state.get("director_log", []) + ["[PAPER_READER] No papers found."],
        }

    summaries = []
    for paper in papers:
        _log(state, "paper_reader", f"Reading: {paper['name']}")
        text = read_local_pdf.invoke({"pdf_path": paper["path"], "max_pages": 25})

        if text.startswith("ERROR"):
            summaries.append({"file": paper["name"], "error": text})
            continue

        # Get bullet-point prompt structure
        bp_json = bullet_point_paper.invoke({"paper_text": text, "focus": state.get("research_focus", "multimodal medical imaging")})
        bp_data = json.loads(bp_json)

        # Send to LLM for actual analysis
        system = (
            "You are an expert biomedical AI researcher with the analytical style of "
            "Andrej Karpathy — precise, first-principles, skeptical of vague claims. "
            "You work in the SMILE Lab under Prof. Ruogu Fang, focused on multimodal "
            "medical imaging (MRI, OCTA, cardiac, retinal). "
            "Be brutally honest about limitations. Use LaTeX for math."
        )
        prompt = f"""
Analyze this paper systematically. Focus domain: {bp_data['focus_domain']}.

Paper excerpt:
{bp_data['paper_excerpt']}

Produce a structured analysis with these sections:
1. **Core Contribution** (1-2 sentences, no fluff)
2. **Methodology** (key architectural choices, loss functions with math if any)
3. **Datasets Used** (name, size, modality, access type)
4. **Key Results** (specific numbers, not vague claims)
5. **Mathematical Formulation** (key equations in LaTeX)
6. **Limitations** (be honest — what does this paper NOT solve?)
7. **Open Research Gaps** (3-5 specific gaps, numbered, with novelty score 1-10)
8. **SMILE Lab Alignment** (how does this connect to REVEAL++ or multimodal fusion?)
"""
        analysis = _llm(state, prompt, system=system)
        summaries.append({
            "file": paper["name"],
            "path": paper["path"],
            "word_count": bp_data["word_count"],
            "analysis": analysis,
        })

    log_entries = state.get("director_log", []) + [
        f"[PAPER_READER] Read and analyzed {len(summaries)} paper(s)."
    ]
    return {"paper_summaries": summaries, "director_log": log_entries}


# ---------------------------------------------------------------------------
# NODE 2: Literature Scout
# ---------------------------------------------------------------------------

def literature_scout_node(state: ResearchDirectorState) -> dict:
    """
    Search ArXiv and OpenAlex for recent SOTA papers relevant to the research focus.
    """
    focus = state.get("research_focus", "multimodal medical imaging MRI OCTA registration")
    queries = [
        f"{focus} deep learning 2024 2025",
        "multimodal MRI OCTA fusion self-supervised",
        "cardiac motion estimation registration transformer",
        "retinal fundus biomarker multimodal AI",
        "REVEAL foundation model medical imaging",
    ]

    all_arxiv = []
    all_openalex = []

    for q in queries[:3]:  # Limit to avoid rate-limiting
        _log(state, "literature_scout", f"ArXiv search: {q[:50]}...")
        arxiv_raw = search_arxiv_papers.invoke({"query": q, "max_results": 5, "year_from": 2023})
        arxiv_data = json.loads(arxiv_raw)
        all_arxiv.extend(arxiv_data.get("results", []))

    for q in queries[3:]:
        _log(state, "literature_scout", f"OpenAlex search: {q[:50]}...")
        oa_raw = search_openalex_papers.invoke({"query": q, "max_results": 4, "filter_year": 2022})
        oa_data = json.loads(oa_raw)
        all_openalex.extend(oa_data.get("results", []))

    # De-duplicate by title
    seen = set()
    unique_arxiv = []
    for p in all_arxiv:
        key = p["title"][:50].lower()
        if key not in seen:
            seen.add(key)
            unique_arxiv.append(p)

    # LLM synthesis of the literature landscape
    lit_summary_prompt = f"""
You are a SMILE Lab research scientist. Synthesize the following recent papers into:

1. **Current SOTA Summary** — what are the leading methods in multimodal medical imaging?
2. **Dominant Trends** (3-5 trends with paper counts backing them)
3. **Underexplored Areas** — what is everyone ignoring?
4. **Best Papers to Deep-Read** — pick top 3 from the list and explain why

ArXiv papers found:
{json.dumps(unique_arxiv[:10], indent=2)}

OpenAlex papers found:
{json.dumps(all_openalex[:6], indent=2)}
"""
    synthesis = _llm(state, lit_summary_prompt)

    log_entries = state.get("director_log", []) + [
        f"[LITERATURE_SCOUT] Found {len(unique_arxiv)} ArXiv + {len(all_openalex)} OpenAlex papers."
    ]
    return {
        "sota_papers": {"arxiv": unique_arxiv, "openalex": all_openalex},
        "literature_synthesis": synthesis,
        "director_log": log_entries,
    }


# ---------------------------------------------------------------------------
# NODE 3: Gap Finder
# ---------------------------------------------------------------------------

def gap_finder_node(state: ResearchDirectorState) -> dict:
    """
    Cross-reference local paper analysis with SOTA to identify research gaps.
    """
    _log(state, "gap_finder", "Analyzing research gaps...")

    # Compile all summaries
    paper_analyses = "\n\n".join(
        f"LOCAL PAPER: {s['file']}\n{s.get('analysis', s.get('error', ''))}"
        for s in state.get("paper_summaries", [])
    )
    sota_synthesis = state.get("literature_synthesis", "")
    sota_papers = state.get("sota_papers", {})

    gap_prompt_raw = find_research_gaps.invoke({
        "paper_summaries": paper_analyses[:4000],
        "domain": "multimodal medical imaging",
        "focus": state.get("research_focus", "MRI + OCTA multimodal fusion"),
    })
    gap_data = json.loads(gap_prompt_raw)

    full_prompt = f"""
You are a SMILE Lab research director + Andrej Karpathy hybrid.

Your task: Identify the MOST PROMISING and NOVEL research gaps.

LOCAL PAPER ANALYSES:
{paper_analyses[:3000]}

CURRENT SOTA SYNTHESIS:
{sota_synthesis[:2000]}

Gap categories to investigate:
{json.dumps(gap_data['gap_categories'], indent=2)}

For each gap, provide:
- Gap title (concise)
- Why it's a gap (what papers do / don't do)
- Novelty score (1-10)
- Feasibility score (1-10) for a 1-2 year PhD project
- Specific research question to ask
- Proposed method sketch (1-3 sentences)
- Which existing dataset could test this (name + access type)
- Alignment with SMILE Lab / REVEAL++ (explicit connection)

Prioritize gaps that are:
1. Computationally feasible on 1-2 GPUs
2. Clinically meaningful (not just benchmark chasing)
3. Novel vs existing literature
4. Connectable to multimodal MRI + OCTA data
"""
    gaps_analysis = _llm(state, full_prompt)

    log_entries = state.get("director_log", []) + [
        "[GAP_FINDER] Research gaps identified and ranked."
    ]
    return {
        "research_gaps": gaps_analysis,
        "director_log": log_entries,
    }


# ---------------------------------------------------------------------------
# NODE 4: Math Innovator
# ---------------------------------------------------------------------------

def math_innovator_node(state: ResearchDirectorState) -> dict:
    """
    Propose concrete mathematical modifications to existing methods.
    Combines Karpathy's first-principles approach with SMILE Lab's research aims.
    """
    _log(state, "math_innovator", "Proposing mathematical innovations...")

    # Seed from existing pipeline math
    existing_methods = """
Current pipeline uses:
1. 3D U-Net deformable registration: φ = f_θ(I_fixed, I_moving)
2. Loss: L = w_ssim * L_SSIM + w_smooth * L_smooth + w_contrast * L_contrastive
3. Spatial Transformer Network (STN) for warping
4. Directional cosine motion descriptor: α_t = cos(θ) where θ = angle(φ_t, r)
5. Multimodal fusion head: [α_t^cardiac, f_octa, f_text] → prediction
"""
    target = state.get("research_focus", "improve multimodal fusion for MRI + OCTA")
    gaps = state.get("research_gaps", "")[:2000]

    math_prompt_raw = propose_math_modification.invoke({
        "existing_method": existing_methods,
        "target_improvement": f"{target}. Identified gaps: {gaps[:500]}",
        "constraints": "differentiable, GPU-compatible, < 50M parameters, clinically interpretable",
    })
    math_data = json.loads(math_prompt_raw)

    full_prompt = f"""
You are a mathematical AI researcher with deep expertise in differential geometry,
information theory, and deep learning optimization.

Apply Karpathy's principles:
{json.dumps(math_data['karpathy_principles'], indent=2)}

EXISTING METHODS:
{math_data['existing_method']}

TARGET IMPROVEMENT:
{math_data['target_improvement']}

CONSTRAINTS:
{math_data['constraints']}

Propose 3 specific mathematical modifications, each including:

### Modification [N]: [Name]
**Intuition**: (1 sentence — what physical/statistical insight drives this)
**Mathematical formulation** (full LaTeX):
$$[equation]$$
**Gradient derivation** (key steps):
**Expected improvement**: [specific metric, e.g. +2% DSC on BraTS]
**Ablation design** (1-2 experiments to validate):
**Failure modes to watch**:
**Related work** (1-2 citations):

Also suggest the most promising combination of these modifications.
"""
    innovations = _llm(state, full_prompt)

    log_entries = state.get("director_log", []) + [
        "[MATH_INNOVATOR] Mathematical modifications proposed."
    ]
    return {
        "mathematical_innovations": innovations,
        "director_log": log_entries,
    }


# ---------------------------------------------------------------------------
# NODE 5: Experiment Runner
# ---------------------------------------------------------------------------

def experiment_runner_node(state: ResearchDirectorState) -> dict:
    """
    Design and optionally run ablation experiments.
    Always designs the matrix; runs only if run_experiments=True in state.
    """
    _log(state, "experiment_runner", "Designing ablation matrix...")

    components = ["ssim_loss", "smoothness_loss", "contrastive_loss", "octa_fusion", "text_features"]
    datasets = ["BraTS2023", "DRIVE_OCTA", "UKBB_cardiac"]  # use sample data if unavailable
    metrics = ["DSC", "HD95", "NCC", "SSIM", "alpha_mean", "alpha_std"]

    ablation_raw = generate_ablation_matrix.invoke({
        "model_components": components,
        "datasets": datasets,
        "metrics": metrics,
        "max_experiments": 12,
    })
    ablation_data = json.loads(ablation_raw)

    experiment_result = None
    if state.get("run_experiments", False):
        _log(state, "experiment_runner", "Running quick experiment (3 epochs)...")
        exp_raw = run_quick_experiment.invoke({
            "config_overrides": {
                "loss.contrastive_weight": 0.0,  # ablate contrastive
                "model.dropout": 0.2,
            },
            "epochs": 3,
            "dataset_path": "data/study_images",
        })
        experiment_result = json.loads(exp_raw)

    # LLM interprets the ablation design
    interp_prompt = f"""
As SMILE Lab director, interpret this ablation matrix and prioritize which
experiments to run first given a 2-GPU, 48-hour compute budget.

Ablation matrix:
{json.dumps(ablation_data['ablation_matrix'][:6], indent=2)}

Total estimated runtime: {ablation_data['estimated_total_runtime_hours']} hours
Quick experiment result: {json.dumps(experiment_result, indent=2) if experiment_result else 'Not run yet'}

Provide:
1. **Priority order** for the ablation experiments (with rationale)
2. **Expected findings** based on prior work
3. **Decision criteria** — what result would make you pivot?
4. **Statistical significance** — how many seeds/folds needed?
"""
    experiment_plan = _llm(state, interp_prompt)

    log_entries = state.get("director_log", []) + [
        f"[EXPERIMENT_RUNNER] Ablation matrix: {ablation_data['total_experiments']} experiments designed."
    ]
    return {
        "ablation_design": ablation_data,
        "experiment_result": experiment_result,
        "experiment_plan": experiment_plan,
        "director_log": log_entries,
    }


# ---------------------------------------------------------------------------
# NODE 6: Results Analyst
# ---------------------------------------------------------------------------

def results_analyst_node(state: ResearchDirectorState) -> dict:
    """
    Interpret experimental results, generate statistical analysis,
    and produce publication-ready figure descriptions.
    """
    _log(state, "results_analyst", "Analyzing results...")

    exp_result = state.get("experiment_result")
    ablation_design = state.get("ablation_design", {})
    innovations = state.get("mathematical_innovations", "")

    analysis_prompt = f"""
You are a rigorous statistical analyst for a biomedical AI paper.

Experimental results:
{json.dumps(exp_result, indent=2) if exp_result else "No experimental results yet — analyze the ablation DESIGN."}

Ablation matrix summary:
{json.dumps(ablation_design.get('ablation_matrix', [])[:4], indent=2)}

Mathematical innovations proposed:
{innovations[:1500]}

Produce:
1. **Statistical Analysis Plan**
   - Required sample sizes for 0.05 significance
   - Recommended statistical tests (paired t-test, Wilcoxon, etc.)
   - Multiple comparison correction (Bonferroni / FDR)

2. **Expected Figure Set** (describe each figure):
   - Figure 1: [title + what it shows + which metric]
   - Figure 2: ...
   - Figure 3: ...

3. **Table Design** — ablation table with columns/rows

4. **Narrative for Results Section** — 2 paragraphs connecting
   experiments to the proposed mathematical modifications

5. **Karpathy Sanity Check** — what trivial baseline must be beaten first?
"""
    results_analysis = _llm(state, analysis_prompt)

    log_entries = state.get("director_log", []) + [
        "[RESULTS_ANALYST] Results analysis and figure plan complete."
    ]
    return {
        "results_analysis": results_analysis,
        "director_log": log_entries,
    }


# ---------------------------------------------------------------------------
# NODE 7: LaTeX Writer
# ---------------------------------------------------------------------------

def latex_writer_node(state: ResearchDirectorState) -> dict:
    """
    Draft key paper sections in LaTeX, grounded in this session's analyses.
    """
    _log(state, "latex_writer", "Drafting LaTeX sections...")

    paper_summaries = state.get("paper_summaries", [])
    gaps = state.get("research_gaps", "")
    innovations = state.get("mathematical_innovations", "")
    results = state.get("results_analysis", "")

    latex_prompt = f"""
You are writing a research paper for submission to MICCAI or IEEE TMI.
Draft the following LaTeX sections based on the research analysis.

Research gaps identified:
{gaps[:1000]}

Mathematical innovations:
{innovations[:1500]}

Results analysis:
{results[:1000]}

Produce three LaTeX sections:

\\section{{Introduction}}
(motivation, clinical significance, gap statement, contribution bullet points)

\\section{{Methodology}}
(architecture description, full loss function derivation with equations,
dataset preprocessing, training protocol)

\\section{{Experiments}}
(ablation table template, datasets table, evaluation metrics justified)

Use proper LaTeX: \\cite{{}}, \\begin{{equation}}, \\label{{}}, \\ref{{}}.
Include placeholder citations like \\cite{{voxelmorph2019}}, \\cite{{smile_reveal_2024}}.
"""
    latex_sections = _llm(state, latex_prompt)

    log_entries = state.get("director_log", []) + [
        "[LATEX_WRITER] LaTeX sections drafted."
    ]
    return {
        "latex_sections": latex_sections,
        "director_log": log_entries,
    }


# ---------------------------------------------------------------------------
# NODE 8: Email Drafter + Sender
# ---------------------------------------------------------------------------

def email_drafter_node(state: ResearchDirectorState) -> dict:
    """
    Draft a professional collaboration email to Prof. Ruogu Fang,
    grounded in this session's paper analysis and research contributions.
    Then optionally send it.
    """
    _log(state, "email_drafter", "Drafting collaboration email...")

    gaps = state.get("research_gaps", "")[:800]
    innovations = state.get("mathematical_innovations", "")[:600]
    paper_analyses = "\n".join(s.get("analysis", "")[:300] for s in state.get("paper_summaries", []))[:800]

    email_prompt_raw = draft_collaboration_email.invoke({
        "subject": "Collaboration Inquiry — Multimodal MRI+OCTA Research for SMILE Lab",
        "context": (
            f"Background: PhD applicant with strong PyTorch background, "
            f"built a 3D U-Net deformable registration pipeline for MRI/OCTA fusion. "
            f"Research gaps I've identified: {gaps[:300]}. "
            f"Mathematical innovations I'm proposing: {innovations[:300]}. "
            f"Paper analysis conducted: {paper_analyses[:200]}."
        ),
        "tone": "professional_enthusiastic",
    })
    email_meta = json.loads(email_prompt_raw)

    # LLM writes the actual email body
    system = (
        "You are a brilliant, motivated PhD applicant writing to Prof. Ruogu Fang. "
        "You have deeply read her REVEAL++ and multimodal imaging papers. "
        "Write concisely — professors are busy. Lead with substance, not flattery."
    )
    email_body_prompt = f"""
Write a professional collaboration inquiry email using this structure:
{json.dumps(email_meta['sections'], indent=2)}

Context about my research:
{email_meta['context']}

Constraints:
{json.dumps(email_meta['constraints'], indent=2)}

Professor profile:
{json.dumps(email_meta['professor_profile'], indent=2)}

Write ONLY the email body (no "Subject:" header — that's handled separately).
Start directly with "Dear Prof. Fang,".
End with "Best regards,\\nKashim Mirza\\n[GitHub: github.com/kashimmirza]"
"""
    email_body = _llm(state, email_body_prompt, system=system)

    # Optionally send
    send_result = None
    if state.get("send_email", False):
        _log(state, "email_drafter", "Sending email via SMTP...")
        send_raw = send_collaboration_email.invoke({
            "subject": email_meta["subject"],
            "body": email_body,
            "sender_name": "Kashim Mirza",
        })
        send_result = json.loads(send_raw)

    log_entries = state.get("director_log", []) + [
        f"[EMAIL_DRAFTER] Email drafted. Send status: {send_result.get('status', 'not_sent') if send_result else 'draft_only'}."
    ]
    return {
        "email_draft": {
            "subject": email_meta["subject"],
            "to": "ruogu.fang@bme.ufl.edu",
            "body": email_body,
            "send_result": send_result,
        },
        "director_log": log_entries,
    }


# ---------------------------------------------------------------------------
# NODE 9: Karpathy Critic (final review)
# ---------------------------------------------------------------------------

def karpathy_critic_node(state: ResearchDirectorState) -> dict:
    """
    Final critique node. Challenges every claim from all previous nodes.
    Acts as a pre-submission peer reviewer using Karpathy's philosophy.
    """
    _log(state, "karpathy_critic", "Applying Karpathy critique...")

    innovations = state.get("mathematical_innovations", "")[:1000]
    gaps = state.get("research_gaps", "")[:800]
    results = state.get("results_analysis", "")[:800]

    critique_raw = karpathy_critique.invoke({
        "claim": "Our multimodal MRI+OCTA fusion with modified loss improves registration",
        "method": f"3D U-Net + STN + modified loss. Innovations: {innovations[:300]}",
        "result": results[:300] if results else "No results yet — evaluate the experimental design",
    })
    critique_data = json.loads(critique_raw)

    full_critique_prompt = f"""
Apply the Karpathy critique framework to our full research pipeline:

Framework:
{json.dumps(critique_data['critique_framework'], indent=2)}

Research gaps we claim to address:
{gaps[:600]}

Mathematical innovations:
{innovations[:600]}

Results/analysis:
{results[:600]}

For EACH critique category, give:
- **Severity**: HIGH / MEDIUM / LOW
- **Specific concern**: (1-2 sentences)
- **How to address it**: (concrete action)

Then give an overall verdict:
- Publishability score: X/10
- Strongest aspect: ...
- Critical weakness to fix before submission: ...
- 3 action items in priority order
"""
    critique = _llm(state, full_critique_prompt)

    log_entries = state.get("director_log", []) + [
        "[KARPATHY_CRITIC] Critique complete. Review findings above."
    ]
    return {
        "karpathy_critique_output": critique,
        "director_log": log_entries,
    }
