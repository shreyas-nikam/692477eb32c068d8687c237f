import pytest
from definition_ae9112a31a7848dda9b3aadb390d64c7 import get_zero_shot_benchmark_scores

@pytest.mark.parametrize("model_family, model_size, expected", [
    # Test Case 1: Standard functionality - Gemma 3 27B-IT
    ("Gemma 3", "27B-IT", {"MMLU-Pro": 67.5, "LiveCodeBench": 29.7, "Bird-SQL (dev)": 54.4, "GPQA Diamond": 42.4, "SimpleQA": 10.0, "FACTS Grounding": 74.9, "Global MMLU-Lite": 75.1, "MATH": 89.0, "HiddenMath": 60.3, "MMMU (val)": 64.9}),

    # Test Case 2: Standard functionality - Gemini 1.5 Pro
    ("Gemini 1.5", "Pro", {"MMLU-Pro": 75.8, "LiveCodeBench": 34.2, "Bird-SQL (dev)": 54.4, "GPQA Diamond": 59.1, "SimpleQA": 24.9, "FACTS Grounding": 80.0, "Global MMLU-Lite": 80.8, "MATH": 86.5, "HiddenMath": 52.0, "MMMU (val)": 65.9}),

    # Test Case 3: Edge case - Model family not found
    ("Unknown Family", "27B-IT", {}),

    # Test Case 4: Edge case - Model size not found for an existing family
    ("Gemma 3", "Unknown Size", {}),
    
    # Test Case 5: Edge case - Invalid input type for model_family (int instead of str)
    # The actual implementation in the notebook uses `model_family.startswith("Gemini")`,
    # which would raise an AttributeError if `model_family` is not a string.
    (123, "27B-IT", AttributeError),
])
def test_get_zero_shot_benchmark_scores(model_family, model_size, expected):
    try:
        result = get_zero_shot_benchmark_scores(model_family, model_size)
        assert result == expected
    except Exception as e:
        assert isinstance(e, expected)