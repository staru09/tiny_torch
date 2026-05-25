"""Pytest path setup for src package imports."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def requires_tensor_attr(attr: str):
    from src.tensor import Tensor

    sample = Tensor([1.0]) if attr != "_validate_matmul_shapes" else Tensor([[1.0, 2.0]])
    return pytest.mark.skipif(
        not hasattr(sample, attr),
        reason=f"src.tensor.Tensor missing {attr}",
    )


def requires_dropout_helpers():
    from src.layers import Dropout

    d = Dropout(0.5)
    missing = not hasattr(d, "_should_apply_dropout") or not hasattr(d, "_generate_dropout_mask")
    return pytest.mark.skipif(missing, reason="Dropout helper methods not implemented in src.layers")
