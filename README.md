
# exp-phases (v3)

This version *vendors your full original python file* and re-exports the exact classes/functions, avoiding any truncation during extraction.

- `original_impl.py` — your full code as provided
- `env.py` / `model.py` / `transforms.py` — thin re-exports from `original_impl.py`
