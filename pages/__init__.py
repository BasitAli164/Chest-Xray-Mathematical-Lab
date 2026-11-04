"""Streamlit multipage modules.

Each file in this directory becomes a page in the sidebar automatically
(named after the file, e.g. `matrix_explorer.py` -> "Matrix Explorer").

Contract
--------
- These are *thin* UI layers: they call `core/*`, `mathematical_engine/*`
  and `visualization/*`, and render results with `st.*`.
- No heavy computation here.

Phase 1 status
--------------
Placeholder package. Real pages added in Phase 2+:
dashboard, matrix_explorer, enhancement, noise_reduction,
edge_detection, pca_analysis, svd_compression,
feature_analysis, reports.
"""