"""Mathematical Execution Trace Engine.

This package records the *actual* computations performed by the core layer
so the UI can display formulas, derivations and intermediate values.

Contract
--------
- Consumes arrays/parameters produced by `core/*`.
- Never recomputes or fabricates values independently.
- Attaches a structured dict (trace) to each operation:
    {
        "operation_id":       str,
        "operation":          str,
        "formula":            str,
        "input_shape":        list[int],
        "input_dtype":        str,
        "parameters":         dict,
        "derivation":         list[dict],
        "intermediate_results": list[dict],
        "output_shape":       list[int],
        "execution_time_ms":  float,
        "validation":         dict,
    }

Phase 1 status
--------------
Placeholder package. Real modules added in Phase 2:
trace_manager, formula_registry, derivation_engine,
matrix_inspector, numerical_calculator, computation_validator,
explanation_templates.
"""