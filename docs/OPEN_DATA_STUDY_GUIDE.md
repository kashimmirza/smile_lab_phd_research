<!-- @format -->

# Open-Source Study Data Guide

This repository now includes a lightweight experimental study scaffold for image-text ablation research. The goal is to support an open-source pipeline that can ingest public medical imaging and paper-text data, then run simple ablation-style experiments.

## Suggested public sources

1. **Kaggle**

   - medical imaging collections such as retinal OCT, chest X-ray, and MRI datasets
   - useful when you want a quick, reproducible starting point for image-only or image-text experiments

2. **BioImage Archive / NIH Open Access**

   - biomedical imaging datasets with accompanying metadata and publication links
   - suitable for pairing images with paper abstracts or related text

3. **GitHub repositories**

   - search for open medical imaging benchmarks, OCTA datasets, and paper-text corpora
   - useful for downloading curated CSV/JSON manifests and raw data

4. **OpenAlex / Semantic Scholar**
   - query paper metadata for relevant studies and extract titles, abstracts, and citation context
   - pairing these with images creates a plausible image-text ablation workflow

## Recommended workflow

1. Download a public image dataset and place it under a local data folder.
2. Download or scrape paper metadata / abstracts into plain text files.
3. Place the image files in a folder such as `data/study_images`.
4. Place the paper text files in `data/study_papers`.
5. Run the experimental pipeline via the orchestrator or the experimental module.

## Minimal folder layout

```text
data/
  study_images/
    sample_001.npy
    sample_002.npy
  study_papers/
    sample_001.txt
    sample_002.txt
```

## Notes on data scarcity

When the dataset is small, the study can be augmented with synthetic perturbations or text expansion strategies. A practical approach is:

- apply intensity jitter, flips, rotations, and noise to images
- generate paraphrased or expanded text summaries from paper abstracts
- use the resulting variants to stress-test multimodal ablation behavior

This is a suitable prototype for an assistantship-ready research story because it shows a reproducible experimental setup even when access to full-scale clinical data is limited.
