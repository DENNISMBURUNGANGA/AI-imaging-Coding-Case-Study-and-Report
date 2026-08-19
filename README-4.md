# Hybrid Biomedical Image-Analysis Pipeline

**7PAM2032 — Data Analysis with AI · Assignment 3**
Dennis Mburu Nganga · 24179558 · MSc Data Science, University of Hertfordshire

A modular pipeline for nuclei analysis in fluorescence microscopy that replaces free-text
image description with an auditable chain:

```
raw image → segmentation mask → region-property quantification → structured JSON → narrative
```

Every number in the final record is derived from a measurement, not from a language model's
impression of an image.

---

## Headline result

The pipeline runs end to end and is **not** trustworthy. That is the finding, not a caveat.

| Metric | Value |
|---|---|
| U-Net macro Dice / IoU (validation) | 0.9961 / 0.9923 |
| Otsu baseline Dice (same images) | 0.9733 |
| Schema-valid JSON records | 12 / 12 |
| LLM count fidelity vs computed table | 100% |
| Count error, dense and clustered fields | −43% to −67% |
| `density_class` accuracy (test) | 50% |
| Corrupted images caught by `quality_flag` | 0 / 4 |

Two results drive the analysis:

1. **The counting error is structural, not methodological.** Counting connected components in a
   *perfect* ground-truth mask still undercounts clustered nuclei by 57.8%. Otsu contributes only
   4.7 percentage points on top of that. Binary segmentation cannot separate touching objects, so
   every downstream number — count, density class, narrative — inherits that ceiling.

2. **Dice is the wrong headline metric here.** Dice measures pixel overlap and is blind to whether
   a contiguous region contains one nucleus or five. A model reaches 0.9961 Dice while undercounting
   by 43% precisely because the mask it is matching carries the same binary defect.

---

## Pipeline stages

| Task | What it does | Key output |
|---|---|---|
| **1** | Data prep, EDA, and VLM prompt engineering against a local vision model | Naive vs structured prompt comparison |
| **2** | Otsu thresholding, morphological cleanup, `regionprops` feature tables | Per-object feature CSV |
| **3** | U-Net trained from scratch, evaluated with Dice/IoU, loss ablation | Trained weights, metric tables |
| **4** | Full hybrid chain across 12 unseen test images | Per-image JSON + aggregated CSV |
| **5** | Report with critical analysis | 4-page PDF |

**Extension:** corruption robustness — four corrupted images traced through all four stages
(image statistics → mask → feature table → narrative).

---

## Notable design decisions

- **Blue-channel extraction instead of luma conversion.** The nuclear signal sits almost entirely
  in the blue channel. ITU-R 601-2 luma (`0.299R + 0.587G + 0.114B`) evaluates to roughly `0.355·I`
  on this modality, discarding about two thirds of the dynamic range. Direct blue-channel extraction
  retains the full 8-bit range and nearly triples contrast (std 17.20 → 48.49) for one line of code.
- **Label-preserving augmentation only** (flips and 90° rotations), so the binary mask is never
  interpolated — at the cost of never exposing the model to intermediate orientations.
- **Structured output contract for the LLM**: fixed keys, closed vocabularies on classification
  fields, a permitted `"uncertain"` value, and a programmatic `n_objects_matches_computed` check that
  re-derives the count from the feature table rather than trusting the model.
- **`qwen2.5vl:7b` substituted for `llama3.2-vision`** — the suggested model could not be loaded
  (mllama architecture failure in the installed Ollama binary).

---

## Known limitations

- The dataset is fully synthetic and lacks staining artefacts, focal drift, debris and real
  biological variability, so the near-saturated Dice is a property of the data more than of the model.
- Sample sizes are small throughout (20 validation images; n = 3–5 for variability and temperature runs).
- U-Net input is scaled by a fixed constant (`÷255`) and **never normalised per image**, so an affine
  intensity shift moves the input outside the distribution its decision boundary was learned in. This
  is why the low-contrast corruption collapsed the mask into a single region covering the whole field —
  and why `quality_flag` still read `"ok"`.

---

## Stack

- Python 3 · PyTorch · scikit-image · NumPy · pandas · Matplotlib
- Local LLM/VLM inference via [Ollama](https://ollama.com) (CPU)

---

## Repository structure

<!-- Adjust these names to match your actual files before pushing. -->

```
.
├── notebooks/          # Task 1–4 notebooks
├── src/                # pipeline modules
├── outputs/
│   ├── json/           # per-image structured records
│   ├── figures/        # figures used in the report
│   └── metrics/        # aggregated CSVs
├── report/             # 4-page PDF report
├── requirements.txt
└── README.md
```

---

## Reproducing

```bash
pip install -r requirements.txt

# pull the vision model (Ollama must be running)
ollama pull qwen2.5vl:7b

# then run the notebooks in order: Task 1 → Task 4
```

---

## Academic context

Submitted coursework for 7PAM2032 at the University of Hertfordshire. Published for portfolio
purposes; please do not submit any part of it as your own work.
