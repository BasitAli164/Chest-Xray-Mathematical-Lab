"""Core computational layer for the Chest X-ray Mathematical Lab.

This package contains pure-Python / NumPy / SciPy / OpenCV functions that
perform all image-processing and linear-algebra operations.

Contract
--------
- MUST NOT import Streamlit or any UI code.
- Every function MUST be unit-testable with pytest.
- Every function MUST be deterministic for a given input.
- Mathematical operations MUST expose intermediate values where the
  mathematical_engine expects them (from Phase 2 onwards).

Phase 1 modules
---------------
- image_loader         : file validation, decoding, grayscale conversion
- image_preprocessing  : dtype/shape normalization, resize helpers
"""