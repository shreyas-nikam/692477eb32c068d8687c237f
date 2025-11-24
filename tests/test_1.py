import pytest
import math
import numpy as np

# definition_e6f51fa96e304496bac908e1786b8680 block
from definition_e6f51fa96e304496bac908e1786b8680 import calculate_memory_footprint

@pytest.mark.parametrize(
    "model_name, quantization_strategy, include_kv_cache, expected",
    [
        # Test Case 1: Standard usage for a common model and quantization strategy
        ("Gemma3-4B-IT", "Int4", True, 7.3),
        # Test Case 2: Smallest model, bfloat16 (no quantization savings), with KV cache
        ("Gemma3-1B", "bfloat16", True, 2.9),
        # Test Case 3: Largest model, SFP8 quantization, without KV cache (demonstrates memory savings)
        ("Gemma3-27B-IT", "SFP8", False, 27.4),
        # Test Case 4: Invalid model name (expected to gracefully return np.nan)
        ("NonExistentGemmaModel", "bfloat16", False, np.nan),
        # Test Case 5: Invalid type for model_name (expected to raise a TypeError when .get() is called on None)
        (None, "Int4", True, TypeError),
    ]
)
def test_calculate_memory_footprint(model_name, quantization_strategy, include_kv_cache, expected):
    try:
        result = calculate_memory_footprint(model_name, quantization_strategy, include_kv_cache)
        if expected is np.nan:
            assert math.isnan(result)
        else:
            assert result == expected
    except Exception as e:
        assert isinstance(e, expected)