import pytest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from unittest.mock import patch, MagicMock
import numpy as np

# Keep the definition_babdaf3e0cb44d5e8ccc389cf8903bf5 block as it is. DO NOT REPLACE or REMOVE the block.
from definition_babdaf3e0cb44d5e8ccc389cf8903bf5 import plot_grouped_bar_chart_memory

@pytest.fixture(autouse=True)
def mock_plotting():
    """Fixture to mock matplotlib.pyplot.show and plt.figure for all tests."""
    with patch('matplotlib.pyplot.show') as mock_show, \
         patch('matplotlib.pyplot.figure', return_value=MagicMock()) as mock_figure, \
         patch('matplotlib.pyplot.gcf', return_value=MagicMock()) as mock_gcf, \
         patch('matplotlib.pyplot.suptitle') as mock_suptitle, \
         patch('matplotlib.pyplot.tight_layout') as mock_tight_layout:
        
        # Mock axes for the loop in plot_grouped_bar_chart_memory
        mock_ax = MagicMock()
        mock_ax.get_title.return_value = 'Quantization Strategy = bfloat16'
        mock_ax.containers = [MagicMock()] # Mock containers to prevent iteration errors
        mock_gcf.return_value.axes = [mock_ax] # Ensure there's at least one ax for the loop

        yield mock_show, mock_figure, mock_gcf, mock_suptitle, mock_tight_layout

def test_plot_grouped_bar_chart_memory_valid_data(mock_plotting):
    """
    Test with a valid DataFrame containing multiple data points.
    Verifies that the function runs without errors and attempts to display a plot.
    """
    df = pd.DataFrame({
        "Model": ["Gemma3-4B-IT", "Gemma3-4B-IT", "Gemma3-27B-IT", "Gemma3-27B-IT", "Gemma3-4B-IT"],
        "Quantization Strategy": ["bfloat16", "Int4", "bfloat16", "Int4", "SFP8"],
        "Include KV Cache": ["No", "Yes", "No", "Yes", "No"],
        "Memory (GB)": [8.0, 7.3, 54.0, 32.8, 4.4]
    })
    
    mock_show, mock_figure, mock_gcf, mock_suptitle, mock_tight_layout = mock_plotting
    plot_grouped_bar_chart_memory(df)
    mock_show.assert_called_once()
    mock_figure.assert_called_once()
    mock_suptitle.assert_called_once()
    mock_tight_layout.assert_called_once()


def test_plot_grouped_bar_chart_memory_empty_df(mock_plotting):
    """
    Test with an empty DataFrame.
    The function should handle this gracefully without raising an exception,
    and still attempt to initialize and show a plot (which would be empty).
    """
    df = pd.DataFrame(columns=["Model", "Quantization Strategy", "Include KV Cache", "Memory (GB)"])
    
    mock_show, mock_figure, mock_gcf, mock_suptitle, mock_tight_layout = mock_plotting
    plot_grouped_bar_chart_memory(df)
    mock_show.assert_called_once()
    mock_figure.assert_called_once()
    mock_suptitle.assert_called_once()
    mock_tight_layout.assert_called_once()


def test_plot_grouped_bar_chart_memory_single_row_df(mock_plotting):
    """
    Test with a DataFrame containing only a single row of data.
    The function should plot this single data point without errors.
    """
    df = pd.DataFrame({
        "Model": ["Gemma3-4B-IT"],
        "Quantization Strategy": ["Int4"],
        "Include KV Cache": ["No"],
        "Memory (GB)": [2.6]
    })
    
    mock_show, mock_figure, mock_gcf, mock_suptitle, mock_tight_layout = mock_plotting
    plot_grouped_bar_chart_memory(df)
    mock_show.assert_called_once()
    mock_figure.assert_called_once()
    mock_suptitle.assert_called_once()
    mock_tight_layout.assert_called_once()


def test_plot_grouped_bar_chart_memory_missing_memory_column(mock_plotting):
    """
    Test case for a DataFrame missing the 'Memory (GB)' column.
    This should raise a KeyError when seaborn tries to access it.
    """
    df = pd.DataFrame({
        "Model": ["Gemma3-4B-IT", "Gemma3-27B-IT"],
        "Quantization Strategy": ["bfloat16", "Int4"],
        "Include KV Cache": ["No", "Yes"]
    })
    
    mock_show, mock_figure, mock_gcf, mock_suptitle, mock_tight_layout = mock_plotting
    with pytest.raises(KeyError, match="'Memory \\(GB\\)'"):
        plot_grouped_bar_chart_memory(df)


def test_plot_grouped_bar_chart_memory_non_numeric_memory(mock_plotting):
    """
    Test case for a DataFrame where the 'Memory (GB)' column contains non-numeric data.
    This should lead to a ValueError or TypeError from the plotting library.
    """
    df = pd.DataFrame({
        "Model": ["Gemma3-4B-IT", "Gemma3-4B-IT"],
        "Quantization Strategy": ["bfloat16", "Int4"],
        "Include KV Cache": ["No", "Yes"],
        "Memory (GB)": [8.0, "invalid_data"]
    })
    
    mock_show, mock_figure, mock_gcf, mock_suptitle, mock_tight_layout = mock_plotting
    with pytest.raises((ValueError, TypeError)):
        plot_grouped_bar_chart_memory(df)