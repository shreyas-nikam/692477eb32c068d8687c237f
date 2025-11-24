import pytest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from unittest.mock import patch, MagicMock

# Keep the definition_789fff5afda14c48b760a4b52cb3c21d block as it is. DO NOT REPLACE or REMOVE the block.
from definition_789fff5afda14c48b760a4b52cb3c21d import plot_grouped_bar_chart_multimodal

@pytest.mark.parametrize("input_df, expected_exception, expected_barplot_args", [
    # Test Case 1: Valid DataFrame with multiple models and benchmarks
    (
        pd.DataFrame({
            'Model': ['Gemma3-4B-IT', 'Gemma3-4B-IT', 'Gemma3-12B-IT', 'Gemma3-12B-IT'],
            'Benchmark': ['DocVQA', 'InfoVQA', 'DocVQA', 'InfoVQA'],
            'Score': [75.8, 50.0, 87.1, 64.9]
        }),
        None,
        {'x': 'Benchmark', 'y': 'Score', 'hue': 'Model'}
    ),
    # Test Case 2: Empty DataFrame with expected columns
    (
        pd.DataFrame(columns=['Model', 'Benchmark', 'Score']),
        None,
        {'x': 'Benchmark', 'y': 'Score', 'hue': 'Model'}
    ),
    # Test Case 3: DataFrame with Missing Required Columns ('Benchmark' is missing)
    (
        pd.DataFrame({'Model': ['A'], 'Score': [10]}),
        KeyError,
        {'x': 'Benchmark', 'y': 'Score', 'hue': 'Model'} # Args are passed, but then KeyError occurs
    ),
    # Test Case 4: Non-DataFrame Input
    (
        "not a dataframe", # Example of invalid input type
        AttributeError, # Accessing df.empty or df['col'] on a string will raise AttributeError
        None # barplot_args not relevant as sns.barplot won't be successfully called
    ),
    # Test Case 5: DataFrame with a Single Model and Single Benchmark, multiple entries
    (
        pd.DataFrame({
            'Model': ['Gemma3-4B-IT', 'Gemma3-4B-IT'],
            'Benchmark': ['DocVQA', 'DocVQA'],
            'Score': [75.8, 76.0]
        }),
        None,
        {'x': 'Benchmark', 'y': 'Score', 'hue': 'Model'}
    ),
])
def test_plot_grouped_bar_chart_multimodal(input_df, expected_exception, expected_barplot_args):
    # Mock plotting functions to prevent actual plot display and check calls
    with patch('matplotlib.pyplot.show') as mock_show, \
         patch('seaborn.barplot') as mock_barplot, \
         patch('matplotlib.pyplot.figure'), \
         patch('matplotlib.pyplot.title') as mock_title, \
         patch('matplotlib.pyplot.xlabel') as mock_xlabel, \
         patch('matplotlib.pyplot.ylabel') as mock_ylabel, \
         patch('matplotlib.pyplot.xticks') as mock_xticks, \
         patch('matplotlib.pyplot.legend') as mock_legend, \
         patch('matplotlib.pyplot.tight_layout') as mock_tight_layout:
        
        if expected_exception:
            # Expect an exception to be raised
            with pytest.raises(expected_exception):
                plot_grouped_bar_chart_multimodal(input_df)
            
            # Verify that plotting functions are NOT called if an exception occurs
            # Special handling for KeyError: sns.barplot *is* called, but raises error internally
            if expected_exception == AttributeError:
                mock_barplot.assert_not_called()
            elif expected_exception == KeyError:
                mock_barplot.assert_called_once()
                args, kwargs = mock_barplot.call_args
                assert kwargs['x'] == expected_barplot_args['x']
                assert kwargs['y'] == expected_barplot_args['y']
                assert kwargs['hue'] == expected_barplot_args['hue']
                pd.testing.assert_frame_equal(kwargs['data'], input_df)

            mock_show.assert_not_called()
            mock_title.assert_not_called()
            mock_xlabel.assert_not_called()
            mock_ylabel.assert_not_called()
            mock_xticks.assert_not_called()
            mock_legend.assert_not_called()
            mock_tight_layout.assert_not_called()
        else:
            # No exception expected, verify normal execution
            plot_grouped_bar_chart_multimodal(input_df)

            # Assert that seaborn.barplot was called with correct arguments
            mock_barplot.assert_called_once()
            args, kwargs = mock_barplot.call_args
            assert kwargs['x'] == expected_barplot_args['x']
            assert kwargs['y'] == expected_barplot_args['y']
            assert kwargs['hue'] == expected_barplot_args['hue']
            pd.testing.assert_frame_equal(kwargs['data'], input_df)

            # Assert that other matplotlib functions were called
            mock_title.assert_called_once_with('Gemma 3 IT Multimodal Performance Comparison (with P&S applied)')
            mock_xlabel.assert_called_once_with('Multimodal Benchmark')
            mock_ylabel.assert_called_once_with('Score (%)')
            mock_xticks.assert_called_once_with(rotation=45, ha='right')
            mock_legend.assert_called_once()
            mock_tight_layout.assert_called_once()
            mock_show.assert_called_once()