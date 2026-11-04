# Data Directory

This directory is intentionally **empty of image files** in the repository.

## Rules

- **Do not commit** chest X-ray datasets here. They are large and licensed.
- **Do not commit** user uploads. They live only in memory per session.
- Reference datasets must be downloaded **on demand** by the user, not bundled.

## NIH ChestX-ray14 (Optional Reference)

The NIH ChestX-ray14 dataset can be used for real-data PCA experiments in
Phase 4 (PCA Analysis page → "My folder" mode).

**Official source:** <https://nihcc.app.box.com/v/ChestXray-NIHCC>

**License:** The NIH Clinical Center provides this dataset free of charge
for research purposes. Consult the official download page for current terms
and citation requirements.

**Size:** ~45 GB (112,120 frontal-view X-rays, 30,805 patients).

**How to use (optional):**

1. Download a subset of images (e.g. 20–50 PNG/JPG files) from the official
   source into a local folder **outside** this repository, for example:
   `D:\datasets\chest_xray_subset\`
2. Open the **PCA Analysis** page.
3. In the sidebar, choose **"My folder"**.
4. Enter the folder path and click **Load folder**.
5. Proceed with PCA as usual.

**Do NOT** copy the images into this `data/` folder. Keep them external so
the repository stays small and legal to distribute.

## Synthetic Reference Stack

Phase 4's default mode generates a deterministic synthetic "chest-like"
image stack (`core/reference_dataset.synthesize_reference_stack`). It
requires **no download** and is reproducible from a seed. Every UI text
clearly labels this data as synthetic.

## What's allowed in this directory

- This README file
- Small documentation snippets (< 10 KB text/JSON) describing datasets
- **Never**: images, binaries, model weights, parquet files

## License notes

The application code is MIT-licensed (see top-level `LICENSE`).
Datasets are licensed separately by their providers — the application
does not redistribute them.