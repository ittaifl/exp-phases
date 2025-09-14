
import importlib.util, pathlib, types

_NEED = {
    "WallFixedFOVc", "rescale_env_with_locations", "get_next_move",
    "masked_copy_noisy", "NextStepRNN", "NormReLU", "HardSigmoid"
}

def _load_symbols():
    p = pathlib.Path(__file__).with_name("original_sanitized.py")
    src = p.read_text()
    ns = {}
    exec(src, ns, ns)
    return {k: ns[k] for k in _NEED if k in ns}

SYMS = _load_symbols()
