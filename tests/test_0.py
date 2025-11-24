import pytest
from definition_ff04bd3d4ad24801abbe9591f92e8ac0 import get_gemma_model_parameters

@pytest.mark.parametrize("model_name, expected_output", [
    # Test case 1: Valid and common model name
    ("Gemma3-4B-IT", {"Vision Encoder Parameters": 417, "Embedding Parameters": 675, "Non-embedding Parameters": 3209}),
    # Test case 2: Another valid model name (edge case for 0 vision parameters)
    ("Gemma3-1B", {"Vision Encoder Parameters": 0, "Embedding Parameters": 302, "Non-embedding Parameters": 698}),
    # Test case 3: Non-existent model name (edge case for lookup failure)
    ("Gemma3-99B-NonExistent", {}),
    # Test case 4: Empty string as model name (edge case for lookup failure)
    ("", {}),
    # Test case 5: Invalid type for model_name (e.g., int, as the function expects a string)
    # The implementation's .get() method will simply not find it and return the default {}
    (12345, {}), 
])
def test_get_gemma_model_parameters(model_name, expected_output):
    """
    Test cases for get_gemma_model_parameters function.
    Covers valid model lookups, non-existent models, empty strings, and invalid input types.
    """
    assert get_gemma_model_parameters(model_name) == expected_output