"""Visualization helpers — thin wrappers over matplotlib / st.image.

Contract
--------
- Pure functions that return matplotlib figures or numpy arrays suitable
  for `st.pyplot` / `st.image`.
- MUST NOT call Streamlit's `st.*` directly (that stays in `pages/*`).
- MUST receive arrays as arguments; never read from session_state.

Phase 5 modules
---------------
- math_workspace  : three-level mathematical transparency renderer
- comparison      : reusable before/after and side-by-side viewers
"""