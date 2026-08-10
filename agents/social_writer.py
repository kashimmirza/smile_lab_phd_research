"""
agents/social_writer.py
--------------------------
Drafts platform-specific social posts summarizing real results from this
session (never fabricated numbers), respecting each platform's practical
length constraints. Most social platforms don't render LaTeX, so equations
are described in plain words/unicode, with an animated-equation image
(see agents/equation_animator.py) as the visual instead.

The character limits below are practical working defaults, not
necessarily each platform's exact current maximum (those do change) --
treat them as starting points and confirm against each platform's current
posting guidelines if precision matters for your use case.
"""

MODEL = "claude-sonnet-5"

PLATFORM_SPECS = {
    "linkedin": {
        "max_chars": 3000,
        "style": (
            "Professional, first-person research update aimed at academics/industry peers. "
            "Explain the finding's significance before the technical detail. If an equation is "
            "central, describe it in words or simple unicode (e.g. 'alpha = cos(theta)') rather "
            "than LaTeX syntax, since LinkedIn won't render LaTeX. End with 2-4 relevant hashtags."
        ),
    },
    "twitter": {
        "max_chars": 280,
        "style": (
            "Punchy, single-idea hook -- lead with the most interesting result or question, not "
            "background. Plain words or simple unicode math only. 1-2 hashtags max."
        ),
    },
    "facebook": {
        "max_chars": 3200,  # ~500 words at typical word length, per user's stated target
        "style": (
            "Conversational, explains the work to a general (non-expert) audience. Use an analogy "
            "for any equation instead of notation. Slightly longer-form than LinkedIn is fine."
        ),
    },
    "instagram": {
        "max_chars": 2200,
        "style": (
            "Visual-first caption -- assumes an animated equation image/video is attached. Short "
            "punchy lines with line breaks for readability. 5-10 relevant hashtags at the end."
        ),
    },
}


def _get_client():
    import anthropic
    return anthropic.Anthropic()


def write_social_post(platform: str, instruction: str, session) -> str:
    platform = platform.lower()
    if platform not in PLATFORM_SPECS:
        raise ValueError(f"Unknown platform '{platform}'. Choose from: {list(PLATFORM_SPECS)}")

    spec = PLATFORM_SPECS[platform]
    client = _get_client()

    metrics_summary = ""
    if session.metrics:
        metrics_summary = (
            f"Real results to draw from (do not alter these numbers): "
            f"final training loss = {session.metrics.get('final_loss')}, "
            f"mean alpha descriptor = {session.metrics.get('descriptor_mean_alpha')}, "
            f"ablation results = {session.metrics.get('ablation_results')}."
        )
    else:
        metrics_summary = "No results are available yet in this session -- describe the method/approach only, do not invent outcomes."

    system = (
        f"You write a single {platform} post about a biomedical AI research project. "
        f"Hard limit: {spec['max_chars']} characters -- count carefully and stay under it. "
        f"Style guidance: {spec['style']} "
        "Never fabricate results, numbers, or claims beyond what's provided. "
        "Output ONLY the post text itself -- no preamble like 'Here is your post', no markdown fences."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1200,
        system=system,
        messages=[{"role": "user", "content": f"{metrics_summary}\n\nInstruction: {instruction}"}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def write_all_platforms(instruction: str, session, platforms=None) -> dict:
    platforms = platforms or list(PLATFORM_SPECS.keys())
    return {p: write_social_post(p, instruction, session) for p in platforms}