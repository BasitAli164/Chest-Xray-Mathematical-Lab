"""Plotting and display helpers.

Contract
--------
- Pure functions that return matplotlib figures or numpy arrays
  suitable for `st.pyplot` / `st.image`.
- MUST NOT call Streamlit's `st.*` directly (that stays in `app.py` and
  `pages/*`).
- MUST receive arrays as arguments; never read from session_state.

Phase 1 status
--------------
Placeholder package. Real modules added in Phase 3:
image_viewer, matrix_viewer, mathematical_viewer,
histogram, pca_plots, comparison.
"""