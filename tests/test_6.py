import pytest
# DO NOT REPLACE or REMOVE the block
from definition_7eab302421914bed9c74d4bf6b1aeb6e import get_pre_trained_ability_scores
# DO NOT REPLACE or REMOVE the block

@pytest.mark.parametrize("model_family, model_size, expected_output", [
    # Test Case 1: Standard functionality for a large Gemma 3 model
    ("Gemma 3", "27B", {"Vision": 85, "Code": 90, "Science": 88, "Factuality": 92, "Reasoning": 95, "Multilingual": 90}),
    # Test Case 2: Standard functionality for a small Gemma 2 model
    ("Gemma 2", "2B", {"Vision": 30, "Code": 40, "Science": 45, "Factuality": 50, "Reasoning": 55, "Multilingual": 35}),
    # Test Case 3: Edge case - Non-existent model family
    ("Gemma X", "27B", {}),
    # Test Case 4: Edge case - Existing model family but non-existent model size for that family
    ("Gemma 3", "1B", {}), # 'Gemma 3' in ability_data does not have a '1B' size for pre-trained abilities.
    # Test Case 5: Edge case - Invalid input type for model_family (should gracefully return an empty dict)
    (None, "27B", {}),
])
def test_get_pre_trained_ability_scores(model_family, model_size, expected_output):
    result = get_pre_trained_ability_scores(model_family, model_size)
    assert result == expected_output

