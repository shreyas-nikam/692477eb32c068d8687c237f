import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def run_page1():
    st.markdown("### Gemma 3 Model Overview: Parameter Counts")
    st.markdown("""
    Understanding the scale of a model is crucial for assessing its computational requirements. The Gemma 3 family offers models ranging from 1 to 27 billion parameters, each with specific components like vision encoders, embedding layers, and non-embedding parameters. These counts directly influence the model's complexity and potential performance.

    As per [2, Table 1], the parameter counts for Gemma 3 models are structured as follows:
    -   **Vision Encoder Parameters:** Parameters specific to the vision component.
    -   **Embedding Parameters:** Parameters for token embeddings.
    -   **Non-embedding Parameters:** The majority of the model's parameters, including transformer layers.

    Let $P_V$ be the Vision Encoder Parameters, $P_E$ be the Embedding Parameters, and $P_{NE}$ be the Non-embedding Parameters. The total parameters $P_T$ for a model are given by:
    """)
    st.latex(r" P_T = P_V + P_E + P_{NE} ")
    st.markdown("All parameter counts are typically expressed in millions.")

    _GEMMA_MODEL_PARAMETERS_DATA = {
        "Gemma3-4B-IT": {
            "Vision Encoder Parameters": 417,
            "Embedding Parameters": 675,
            "Non-embedding Parameters": 3209
        },
        "Gemma3-1B": {
            "Vision Encoder Parameters": 0,
            "Embedding Parameters": 302,
            "Non-embedding Parameters": 698
        },
        "Gemma3-12B-IT": {"Vision Encoder Parameters": 417, "Embedding Parameters": 1012, "Non-embedding Parameters": 10759},
        "Gemma3-27B-IT": {"Vision Encoder Parameters": 417, "Embedding Parameters": 1416, "Non-embedding Parameters": 25600},
    }

    def get_gemma_model_parameters(model_name: str) -> dict:
        """
        Retrieves parameter counts for a specified Gemma 3 model from a predefined dataset.
        Returns an empty dict if the model is not found.
        """
        return _GEMMA_MODEL_PARAMETERS_DATA.get(model_name, {})

    gemma_models = list(_GEMMA_MODEL_PARAMETERS_DATA.keys())

    model_parameters = []
    for model in gemma_models:
        params = get_gemma_model_parameters(model)
        if params:
            params['Model'] = model
            model_parameters.append(params)

    model_parameters_df = pd.DataFrame(model_parameters)
    model_parameters_df['Total Parameters (M)'] = model_parameters_df[[
        "Vision Encoder Parameters", "Embedding Parameters", "Non-embedding Parameters"
    ]].sum(axis=1)

    st.markdown("Gemma 3 Model Parameter Counts (in Millions):")
    st.dataframe(model_parameters_df.set_index('Model'))

    st.markdown("""
    The parameter counts for the Gemma 3 models have been retrieved and aggregated into a DataFrame. We can observe the distinct parameter distribution across different model sizes, with the 1B model notably lacking a Vision Encoder component, while larger models share the same Vision Encoder but scale up significantly in Embedding and Non-embedding parameters. This data is foundational for understanding the computational complexity of each model.
    """)

    st.markdown("### Visualizing Model Parameter Counts")
    st.markdown("""
    A visual representation of the parameter counts provides an immediate understanding of the relative size and complexity of each Gemma 3 model. This is especially useful for Financial Data Engineers when considering the hardware capacity required for deployment.
    """)

    def plot_bar_chart(df: pd.DataFrame, x_col: str, y_cols: list, title: str, x_label: str, y_label: str):
        """
        Generates a stacked bar chart of model parameter counts using Matplotlib for Streamlit.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        df.set_index(x_col)[y_cols].plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis', ax=ax)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.xticks(rotation=45, ha='right')
        ax.legend(title='Parameter Type')
        plt.tight_layout()
        st.pyplot(fig)

    st.write("---")
    st.subheader("Gemma 3 Model Parameter Distribution")
    
    if 'selected_models_for_plot' not in st.session_state:
        st.session_state.selected_models_for_plot = ["Gemma3-1B", "Gemma3-4B-IT", "Gemma3-12B-IT", "Gemma3-27B-IT"]

    selected_models_for_plot = st.multiselect(
        "Select models to visualize parameter counts:",
        options=gemma_models,
        default=st.session_state.selected_models_for_plot,
        key="page1_model_selection"
    )
    st.session_state.selected_models_for_plot = selected_models_for_plot

    if selected_models_for_plot:
        plot_bar_chart(
            model_parameters_df[model_parameters_df['Model'].isin(selected_models_for_plot)],
            x_col='Model',
            y_cols=["Vision Encoder Parameters", "Embedding Parameters", "Non-embedding Parameters"],
            title='Gemma 3 Model Parameter Counts by Component (Millions)',
            x_label='Gemma 3 Model',
            y_label='Parameters (Millions)'
        )
    else:
        st.info("Please select at least one model to visualize parameter counts.")

    st.markdown("""
    The stacked bar chart clearly illustrates how the total parameter count scales with model size, and how the distribution across vision encoder, embedding, and non-embedding components changes (or remains constant for the vision encoder across 4B, 12B, 27B models). This visualization highlights the architectural similarities and scaling differences, informing resource allocation for different model sizes.
    """)