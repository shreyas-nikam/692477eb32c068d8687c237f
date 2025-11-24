import pytest
import pandas as pd
from definition_add3a877be574d5f8b1f8274dfe7726e import get_chatbot_arena_elo_scores

@pytest.mark.parametrize(
    "model_list, expected_df, expected_exception",
    [
        # Test Case 1: Valid list of models, ensuring correct scores and sorting
        (["Gemma-3-27B-IT", "Gemini-1.5-Pro-002", "Gemini-2.0-Pro-Exp-02-05"],
         pd.DataFrame([
             {"Model": "Gemini-2.0-Pro-Exp-02-05", "Elo Score": 1380},
             {"Model": "Gemma-3-27B-IT", "Elo Score": 1338},
             {"Model": "Gemini-1.5-Pro-002", "Elo Score": 1302}
         ]), None),
        
        # Test Case 2: Empty list of models - should return an empty DataFrame with correct columns
        ([], pd.DataFrame(columns=['Model', 'Elo Score']), None),
        
        # Test Case 3: List with models not present in the predefined dataset
        (["Unknown-Model-X", "Another-Non-Existent"], pd.DataFrame(columns=['Model', 'Elo Score']), None),
        
        # Test Case 4: Mixed list with some valid and some invalid models
        (["Gemma-3-27B-IT", "Random-Model-123", "Gemma-2-27B-IT", "GPT-4.5-Preview"],
         pd.DataFrame([
             {"Model": "GPT-4.5-Preview", "Elo Score": 1411},
             {"Model": "Gemma-3-27B-IT", "Elo Score": 1338},
             {"Model": "Gemma-2-27B-IT", "Elo Score": 1220}
         ]), None),
        
        # Test Case 5: Invalid input type (e.g., integer instead of list)
        (123, None, TypeError)
    ]
)
def test_get_chatbot_arena_elo_scores(model_list, expected_df, expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            get_chatbot_arena_elo_scores(model_list)
    else:
        result_df = get_chatbot_arena_elo_scores(model_list)
        
        # Ensure the result is a Pandas DataFrame
        assert isinstance(result_df, pd.DataFrame)
        
        # For non-exception cases, compare the DataFrame structure and content
        pd.testing.assert_frame_equal(result_df, expected_df.reset_index(drop=True))