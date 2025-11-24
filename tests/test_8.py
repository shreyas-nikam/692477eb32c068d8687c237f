import pytest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from unittest.mock import patch, MagicMock

# Keep the definition_2e46df48bdd644069a61675a39c6e8fe block as it is. DO NOT REPLACE or REMOVE the block.
from definition_2e46df48bdd644069a61675a39c6e8fe import plot_grouped_bar_chart

@pytest.mark.parametrize(
    "df_input, x_col, y_col, hue_col, title, x_label, y_label, expected_exception, test_description",
    [
        # Test Case 1: Standard successful plot generation
        (
            pd.DataFrame({
                'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
                'Subcategory': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
                'Value': [10, 15, 20, 25, 30, 35]
            }),
            'Category', 'Value', 'Subcategory',
            'Successful Grouped Bar Chart', 'Main Category', 'Measured Value',
            None, "Successful plot generation with valid data."
        ),
        # Test Case 2: Empty DataFrame - should run without error, producing an empty plot
        (
            pd.DataFrame({'Category': [], 'Subcategory': [], 'Value': []}),
            'Category', 'Value', 'Subcategory',
            'Empty Data Plot', 'Categories', 'Values',
            None, "Plotting with an empty DataFrame (should not crash)."
        ),
        # Test Case 3: Missing x_col (or any required column for plotting) - expected KeyError
        (
            pd.DataFrame({'Col1': [1,2], 'Value': [10,20], 'Hue': ['A','B']}),
            'NonExistentCol', 'Value', 'Hue',
            'Missing X Column', 'X-axis', 'Y-axis',
            KeyError, "DataFrame is missing a required column (KeyError expected)."
        ),
        # Test Case 4: Non-DataFrame `df` input (e.g., list) - expected AttributeError
        (
            [1, 2, 3], # Invalid type for df
            'Category', 'Value', 'Subcategory',
            'Invalid DF Type', 'Categories', 'Values',
            AttributeError, "Input 'df' is not a pandas DataFrame (AttributeError expected)."
        ),
        # Test Case 5: y_col with non-numeric data - seaborn converts to NaN, no direct exception from function
        (
            pd.DataFrame({'Category': ['A', 'B'], 'Value': ['ten', 'twenty'], 'Subcategory': ['X', 'Y']}),
            'Category', 'Value', 'Subcategory',
            'Non-numeric Y Plot', 'Categories', 'Non-numeric Values',
            None, "Y-column has non-numeric data (seaborn handles NaNs gracefully)."
        ),
    ]
)
@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.ylabel')
@patch('matplotlib.pyplot.xlabel')
@patch('matplotlib.pyplot.title')
@patch('matplotlib.pyplot.xticks')
@patch('matplotlib.pyplot.legend')
@patch('matplotlib.pyplot.tight_layout')
@patch('matplotlib.pyplot.figure')
@patch('seaborn.barplot')
def test_plot_grouped_bar_chart(
    mock_seaborn_barplot,
    mock_plt_figure,
    mock_plt_tight_layout,
    mock_plt_legend,
    mock_plt_xticks,
    mock_plt_title,
    mock_plt_xlabel,
    mock_plt_ylabel,
    mock_plt_show,
    df_input, x_col, y_col, hue_col, title, x_label, y_label, expected_exception, test_description
):
    """
    Tests the plot_grouped_bar_chart function covering expected functionality and edge cases.
    Mocks matplotlib and seaborn functions to assert their calls without displaying plots.
    """
    if expected_exception:
        with pytest.raises(expected_exception):
            plot_grouped_bar_chart(df_input, x_col, y_col, hue_col, title, x_label, y_label)
        # Ensure no plotting functions were called if an exception occurred early
        mock_seaborn_barplot.assert_not_called()
        mock_plt_figure.assert_not_called()
        mock_plt_show.assert_not_called()
    else:
        # For cases where no exception is expected (successful plots or graceful handling)
        plot_grouped_bar_chart(df_input, x_col, y_col, hue_col, title, x_label, y_label)

        # Assert that core plotting functions were called
        mock_plt_figure.assert_called_once()
        mock_seaborn_barplot.assert_called_once_with(
            x=x_col, y=y_col, hue=hue_col, data=df_input, palette='tab10'
        )
        mock_plt_title.assert_called_once_with(title)
        mock_plt_xlabel.assert_called_once_with(x_label)
        mock_plt_ylabel.assert_called_once_with(y_label)
        mock_plt_xticks.assert_called_once_with(rotation=45, ha='right')
        mock_plt_legend.assert_called_once_with(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        mock_plt_tight_layout.assert_called_once()
        mock_plt_show.assert_called_once()