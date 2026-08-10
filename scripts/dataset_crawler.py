"""
dataset_crawler.py
-------------------
Given a paper (title, DOI, arXiv ID, or local PDF), this script:
  1. Fetches paper metadata + abstract/full text via Semantic Scholar / OpenAlex.
  2. Scans the text for known medical-imaging dataset names.
  3. Resolves each match against a source registry (Kaggle / TCIA / OpenNeuro / PhysioNet / gated).
  4. Auto-downloads from open APIs where possible; flags gated datasets (UK Biobank, ADNI, etc.)
     for manual access application.
  5. Writes a manifest.json that dataset_mri.py / dataset_octa.py can consume.

Usage:
    python scripts/dataset_crawler.py --title "Deep learning for OCTA vessel segmentation"
    python scripts/dataset_crawler.py --doi 10.1000/xyz123
    python scripts/dataset_crawler.py --pdf ./papers/some_paper.pdf

Requires (add to .env / environment):
    S2_API_KEY               - Semantic Scholar API key
    OPENALEX_MAIL_ADDRESS    - polite-pool email for OpenAlex
    KAGGLE_USERNAME / KAGGLE_KEY   - for Kaggle downloads
    TCIA_API_KEY (optional)  - most TCIA collections are open, some need a key
"""

import os
import re
import json
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, field

import requests

DATA_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/dataset_manifest.json")

# ---------------------------------------------------------------------------
# 1. Dataset registry — extend this as you encounter new datasets in papers.
#    "source" tells the fetcher which downloader to use.
#    "gated": True means it needs a manual data-use agreement / application.
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "adni":        {"source": "gated",     "modality": "mri",  "note": "Requires ADNI Data Use Agreement + approval."},
    "uk biobank":  {"source": "gated",     "modality": "mri",  "note": "Requires UK Biobank access application."},
    "brats":       {"source": "kaggle",    "modality": "mri",  "slug": "awsaf49/brats2020-training-data"},
    "isic":        {"source": "url",       "modality": "octa", "url": "https://api.isic-archive.com/api/v2/images/download/"},
    "chexpert":    {"source": "gated",     "modality": "octa", "note": "Requires Stanford ML Group registration."},
    "mimic-cxr":   {"source": "physionet", "modality": "octa", "project": "mimic-cxr/2.0.0"},
    "ixi":         {"source": "url",       "modality": "mri",  "url": "https://brain-development.org/ixi-dataset/"},
    "abide":       {"source": "url",       "modality": "mri",  "url": "http://fcon_1000.projects.nitrc.org/indi/abide/"},
    "drive":       {"source": "kaggle",    "modality": "octa", "slug": "andrewmvd/drive-digital-retinal-images"},
    "stare":       {"source": "url",       "modality": "octa", "url": "https://cecas.clemson.edu/~ahoover/stare/"},
    "oasis":       {"source": "kaggle",    "modality": "mri",  "slug": "ninadaithal/imagesoasis"},
    "isles":       {"source": "url",       "modality": "mri",  "url": "https://www.isles-challenge.org/"},
    "openneuro":   {"source": "openneuro", "modality": "mri",  "note": "resolve accession number from paper text (e.g. ds000123)"},
}

DATASET_NAME_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in DATASET_REGISTRY.keys()) + r")\b",
    re.IGNORECASE,
)
OPENNEURO_ACCESSION_PATTERN = re.compile(r"\bds\d{6}\b")


@dataclass
class PaperContext:
    title: str = ""
    abstract: str = ""
    full_text: str = ""
    matches: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Fetch paper metadata
# ---------------------------------------------------------------------------
def fetch_by_title(title: str) -> PaperContext:
    api_key = os.environ.get("S2_API_KEY", "")
    headers = {"x-api-key": api_key} if api_key else {}
    resp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": title, "fields": "title,abstract,openAccessPdf"},
        headers=headers,
        timeout=20,
    )
    resp.raise_for_status()
    results = resp.json().get("data", [])
    if not results:
        raise ValueError(f"No paper found for title: {title}")
    top = results[0]
    ctx = PaperContext(title=top.get("title", ""), abstract=top.get("abstract") or "")

    pdf_info = top.get("openAccessPdf")
    if pdf_info and pdf_info.get("url"):
        ctx.full_text = try_fetch_pdf_text(pdf_info["url"])
    return ctx


def fetch_by_doi(doi: str) -> PaperContext:
    mailto = os.environ.get("OPENALEX_MAIL_ADDRESS", "")
    resp = requests.get(
        f"https://api.openalex.org/works/https://doi.org/{doi}",
        params={"mailto": mailto} if mailto else {},
        timeout=20,
    )
    resp.raise_for_status()
    work = resp.json()
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    return PaperContext(title=work.get("title", ""), abstract=abstract or "")


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    positions = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def try_fetch_pdf_text(pdf_url: str) -> str:
    """Best-effort PDF text pull. Requires `pip install pypdf`."""
    try:
        import io
        from pypdf import PdfReader

        r = requests.get(pdf_url, timeout=30)
        reader = PdfReader(io.BytesIO(r.content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"  [warn] could not extract PDF text: {e}")
        return ""


def fetch_by_pdf(path: str) -> PaperContext:
    from pypdf import PdfReader

    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    title = text.strip().splitlines()[0] if text.strip() else Path(path).stem
    return PaperContext(title=title, full_text=text)


# ---------------------------------------------------------------------------
# 3. Extract dataset mentions
# ---------------------------------------------------------------------------
def extract_dataset_mentions(ctx: PaperContext) -> list:
    blob = f"{ctx.title}\n{ctx.abstract}\n{ctx.full_text}"
    found = {m.group(1).lower() for m in DATASET_NAME_PATTERN.finditer(blob)}

    accession = OPENNEURO_ACCESSION_PATTERN.search(blob)
    matches = []
    for name in found:
        entry = dict(DATASET_REGISTRY[name])
        entry["name"] = name
        if name == "openneuro" and accession:
            entry["accession"] = accession.group(0)
        matches.append(entry)
    return matches


# ---------------------------------------------------------------------------
# 4. Fetch resolved datasets
# ---------------------------------------------------------------------------
def fetch_dataset(entry: dict) -> dict:
    name = entry["name"]
    out_dir = DATA_DIR / name.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {"name": name, "path": str(out_dir), "status": "unknown"}

    try:
        if entry["source"] == "gated":
            status["status"] = "manual_action_required"
            status["note"] = entry.get("note", "Requires manual application/DUA.")

        elif entry["source"] == "kaggle":
            # Requires `pip install kaggle` and KAGGLE_USERNAME/KAGGLE_KEY set.
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", entry["slug"], "-p", str(out_dir), "--unzip"],
                check=True,
            )
            status["status"] = "downloaded"

        elif entry["source"] == "physionet":
            # PhysioNet restricted projects need credentialed wget with your PhysioNet login.
            status["status"] = "manual_action_required"
            status["note"] = (
                f"Run: wget -r -N -c -np --user <physionet_user> --ask-password "
                f"https://physionet.org/files/{entry['project']}/ -P {out_dir}"
            )

        elif entry["source"] == "openneuro":
            accession = entry.get("accession")
            if accession:
                subprocess.run(
                    ["datalad", "install", f"///openneuro/{accession}"],
                    cwd=out_dir, check=True,
                )
                status["status"] = "downloaded"
            else:
                status["status"] = "accession_not_found"

        elif entry["source"] == "url":
            status["status"] = "manual_action_required"
            status["note"] = f"No public API — visit {entry['url']} to request/download."

    except subprocess.CalledProcessError as e:
        status["status"] = "download_failed"
        status["note"] = str(e)

    return status


# ---------------------------------------------------------------------------
# 5. Manifest + dataloader hookup
# ---------------------------------------------------------------------------
def write_manifest(paper_title: str, results: list):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
    manifest[paper_title] = results
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest updated: {MANIFEST_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Resolve and fetch datasets referenced in a paper.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--title", help="Paper title to search")
    group.add_argument("--doi", help="Paper DOI")
    group.add_argument("--pdf", help="Path to local PDF")
    args = parser.parse_args()

    if args.title:
        ctx = fetch_by_title(args.title)
    elif args.doi:
        ctx = fetch_by_doi(args.doi)
    else:
        ctx = fetch_by_pdf(args.pdf)

    print(f"Resolved paper: {ctx.title}")
    matches = extract_dataset_mentions(ctx)

    if not matches:
        print("No known datasets detected. Add the dataset name to DATASET_REGISTRY and re-run.")
        return

    print(f"Found {len(matches)} candidate dataset(s): {[m['name'] for m in matches]}")
    results = [fetch_dataset(m) for m in matches]

    for r in results:
        print(f"  - {r['name']}: {r['status']}" + (f" ({r['note']})" if "note" in r else ""))

    write_manifest(ctx.title or "unknown_paper", results)


if __name__ == "__main__":
    main()