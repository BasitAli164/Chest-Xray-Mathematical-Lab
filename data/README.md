# Data Directory

This folder is intentionally **empty of image files** in the repository.

## Rules

- **Do not commit** chest X-ray datasets. They are large and licensed.
- **Do not commit** user uploads. They live only in memory per session.
- Reference datasets (e.g. a small NIH ChestX-ray14 subset for PCA in Phase 4)
  must be downloaded on demand from the official source:
  <https://nihcc.app.box.com/v/ChestXray-NIHCC>
- Document any dataset usage in the relevant module's README.

## Phase status

Phase 1 uses **only** user-supplied uploads. No dataset needed.