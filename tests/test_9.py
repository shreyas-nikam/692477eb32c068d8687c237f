import pytest
import pandas as pd
import matplotlib.pyplot as plt
import math
from definition_ee10f8a59e114173a35e1a3f065bbebb import plot_radar_chart

@pytest.mark.parametrize("df_input, categories_input, title_input, expected_exception, exception_match", [
    # Test Case 1: Normal functioning with multiple models and categories
    (pd.DataFrame({'Model': ['M1', 'M2'], 'C1': [80, 70], 'C2': [90, 60], 'C3': [75, 85]}),
     ['C1', 'C2', 'C3'], "Multi-Model Chart", None, None),

    # Test Case 2: Edge Case - Empty DataFrame
    (pd.DataFrame(columns=['Model', 'C1', 'C2', 'C3']),
     ['C1', 'C2', 'C3'], "Empty Data Chart", None, None),

    # Test Case 3: Edge Case - Single Model
    (pd.DataFrame({'Model': ['M1'], 'C1': [80], 'C2': [90], 'C3': [75]}),
     ['C1', 'C2', 'C3'], "Single Model Chart", None, None),

    # Test Case 4: Error Case - Missing categories in DataFrame
    (pd.DataFrame({'Model': ['M1'], 'C1': [80]}),
     ['C1', 'C2'], "Missing Category Chart", KeyError, "not in index|'C2'"),

    # Test Case 5: Error Case - Invalid DataFrame type (e.g., None instead of DataFrame)
    (None, # Invalid df
     ['C1'], "Invalid DF Type", AttributeError, "object has no attribute 'iterrows'"),
])
def test_plot_radar_chart(mocker, df_input, categories_input, title_input, expected_exception, exception_match):
    # Mock matplotlib functions to prevent actual plotting and capture calls
    mock_plt_show = mocker.patch('matplotlib.pyplot.show')
    mocker.patch('matplotlib.pyplot.figure')
    # Mock subplots to return mock Figure and Axes objects
    mock_subplots_return = (mocker.MagicMock(spec=plt.Figure), mocker.MagicMock(spec=plt.Axes))
    mock_plt_subplots = mocker.patch('matplotlib.pyplot.subplots', return_value=mock_subplots_return)
    mocker.patch('matplotlib.pyplot.xticks')
    mocker.patch('matplotlib.pyplot.yticks')
    mocker.patch('matplotlib.pyplot.ylim')
    mocker.patch('matplotlib.pyplot.title')
    mocker.patch('matplotlib.pyplot.legend')

    mock_ax = mock_subplots_return[1] # The mocked Axes object

    if expected_exception:
        with pytest.raises(expected_exception, match=exception_match):
            plot_radar_chart(df_input, categories_input, title_input)
        mock_plt_show.assert_not_called() # show should not be called on error
    else:
        plot_radar_chart(df_input, categories_input, title_input)
        mock_plt_show.assert_called_once()
        mocker.patch('matplotlib.pyplot.title').assert_called_once_with(title_input, size=16, color='black', y=1.1)
        mocker.patch('matplotlib.pyplot.xticks').assert_called_once()
        mocker.patch('matplotlib.pyplot.yticks').assert_called_once()
        mocker.patch('matplotlib.pyplot.ylim').assert_called_once_with(0, 100)
        
        # Check if ax.plot and ax.fill were called only for non-empty DataFrames
        if not df_input.empty:
            assert mock_ax.plot.called
            assert mock_ax.fill.called
        else:
            assert not mock_ax.plot.called
            assert not mock_ax.fill.called