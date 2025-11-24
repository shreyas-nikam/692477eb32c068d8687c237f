import pytest
from definition_56285509477c46c09b0e70a500a01ace import describe_quantization_strategy

@pytest.mark.parametrize("strategy_name, expected", [
    ("bfloat16", "Standard 16-bit floating-point precision, often used for raw model weights."),
    ("Int4", "4-bit integer quantization, drastically reduces memory but may impact precision."),
    ("KV Cache Status (Yes)", "Memory used for model weights *and* the Key-Value cache."),
    ("UnknownStrategy", "Unknown quantization strategy."),
    ("", "Unknown quantization strategy."),
    (None, TypeError),
    (123, TypeError),
])
def test_describe_quantization_strategy(strategy_name, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            describe_quantization_strategy(strategy_name)
    else:
        assert describe_quantization_strategy(strategy_name) == expected