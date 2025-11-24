import pytest
from definition_6602be96cb004e539c7d3e419eb5d192 import get_multimodal_performance_scores

@pytest.mark.parametrize("model_name_param, expected", [
    ("Gemma3-27B-IT", {"MMMU (val)": 64.9, "DocVQA": 86.6, "InfoVQA": 70.6, "TextVQA": 65.1, "AI2D": 84.5, "ChartQA": 78.0, "VQAv2 (val)": 71.0, "MathVista (testmini)": 67.6}),
    ("Gemma3-4B-IT", {"MMMU (val)": 48.8, "DocVQA": 75.8, "InfoVQA": 50.0, "TextVQA": 57.8, "AI2D": 74.8, "ChartQA": 68.8, "VQAv2 (val)": 62.4, "MathVista (testmini)": 50.0}),
    ("NonExistentModel", {}),
    ("", {}),
    (None, TypeError),
])
def test_get_multimodal_performance_scores(model_name_param, expected):
    try:
        result = get_multimodal_performance_scores(model_name_param)
        assert result == expected
    except Exception as e:
        assert isinstance(e, expected)

