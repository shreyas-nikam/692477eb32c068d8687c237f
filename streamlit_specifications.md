
# Streamlit Application Specification: GemmaVision-QuantAdvisor

This document outlines the requirements for developing a Streamlit application based on the provided Jupyter Notebook content and user requirements for Financial Data Engineers. The application, named GemmaVision-QuantAdvisor, will facilitate the evaluation and comparison of Gemma 3 models, focusing on multimodal capabilities, quantization strategies, and performance benchmarks.

## 1. Application Overview

The GemmaVision-QuantAdvisor Streamlit application will serve as an interactive platform for Financial Data Engineers. It will enable users to explore, evaluate, and compare Gemma 3 models and their predecessors, particularly in the context of financial document understanding. The application will highlight architectural characteristics, memory optimizations through quantization, and various performance benchmarks to aid in informed model deployment decisions.

### Learning Goals

Upon interacting with this application, users will be able to:
-   Understand the architectural and performance characteristics of Gemma 3 models.
-   Evaluate the impact of different quantization techniques on memory footprint and efficiency.
-   Compare Gemma 3's performance in multimodal tasks, general intelligence, math, and reasoning against previous Gemma versions and other state-of-the-art models.
-   Utilize quantitative benchmarks and visualizations to guide strategic model deployment in financial data engineering workflows.

## 2. User Interface Requirements

The application will feature a clear and intuitive interface, designed for Financial Data Engineers to easily navigate and interact with complex model data.

### Layout and Navigation Structure

-   **Sidebar:** Will host primary navigation and input controls, including model selection, document upload, and configuration options.
-   **Main Content Area:** Will display model overviews, interactive visualizations, and detailed benchmark results.
-   **Section Headers:** Clearly delineate logical sections such for "Model Overview," "Quantization Strategies," "Multimodal Document Understanding," and "Performance Benchmarks."

### Input Widgets and Controls

-   **Model Selection:**
    -   `st.multiselect` or `st.selectbox` for selecting one or more Gemma 3 models (e.g., "Gemma3-1B", "Gemma3-4B-IT", "Gemma3-12B-IT", "Gemma3-27B-IT") for parameter comparison and benchmarking.
-   **Quantization Strategy Selection:**
    -   `st.radio` buttons or `st.multiselect` for choosing quantization types (e.g., "bfloat16", "Int4", "SFP8").
    -   `st.checkbox` for toggling "KV Caching" (Yes/No) to evaluate memory footprint with and without it.
-   **Document Upload:**
    -   `st.file_uploader` for users to upload financial documents (e.g., scanned annual reports, invoices, charts) in image (PNG, JPG) or PDF formats. This will simulate multimodal input.
-   **Benchmark Selection:**
    -   `st.multiselect` or `st.checkbox` group for selecting specific performance benchmarks to display (e.g., "Chatbot Arena Elo Scores", "Zero-shot General Abilities", "Multimodal Performance", "Long Context Performance", "STEM and Code Performance").
-   **Action Buttons:**
    -   `st.button` to "Run Analysis" after document upload and model/quantization selection.
    -   `st.button` to "Generate Visualizations" for selected benchmarks.

### Visualization Components (Charts, Graphs, Tables)

-   **Model Parameter Counts:**
    -   **Table:** Display of "Gemma 3 Model Parameter Counts (in Millions)" (from `model_parameters_df`).
    -   **Stacked Bar Chart:** Visualizing "Gemma 3 Model Parameter Counts by Component (Millions)" (using `plot_bar_chart`), showing "Vision Encoder Parameters", "Embedding Parameters", and "Non-embedding Parameters" for selected models.
-   **Memory Footprint Comparison:**
    -   **Table:** Display data from "Table 3 | Memory footprints (in GB) comparison between raw (bfloat16) and quantized checkpoints for weights and KV caching (+KV) at 32,768 context size, quantized in 8 bits" from the technical report.
    -   **Grouped Bar Chart:** Comparing memory usage (in GB) across selected models and quantization strategies, with options for including/excluding KV caching.
-   **Performance Benchmarks:**
    -   **Chatbot Arena Elo Scores:**
        -   **Table:** Display "Table 5 | Evaluation of Gemma 3 27B IT model in the Chatbot Arena" from the technical report, potentially filtered.
        -   **Bar Chart:** Visualizing Elo scores for Gemma 3 models against competitive models.
    -   **Zero-shot General Abilities:**
        -   **Table:** Display relevant data from "Table 6 | Performance of instruction fine-tuned (IT) models compared to Gemini 1.5, Gemini 2.0, and Gemma 2 on zero-shot benchmarks across different abilities" from the technical report.
        -   **Radar Charts:** Replicating "Figure 2 | Summary of the performance of different pre-trained models from Gemma 2 and 3 across general abilities," comparing selected Gemma 2 and Gemma 3 models across categories like Vision, Code, Science, Factuality, Reasoning, Multilingual.
    -   **Other Benchmark Tables:** Display selected tables from the technical report (e.g., Tables 9, 10, 11, 12, 13, 14, 15, 16, 17, 18) based on user selections, presenting scores for various tasks and models.
    -   **Long Context Performance:**
        -   **Line Chart:** Depicting perplexity changes with context length for pre-trained and instruction fine-tuned models, similar to "Figure 7 | Long context performance of pre-trained models before and after RoPE rescaling."

### Interactive Elements and Feedback Mechanisms

-   **Dynamic Updates:** All charts, tables, and derived metrics should update in real-time as users change model selections, quantization strategies, or upload documents.
-   **Progress Indicators:** `st.spinner` or `st.progress` will provide feedback during document processing (simulated or real) and benchmark calculations.
-   **Clear Explanations:** Markdown cells will be rendered throughout the application to provide contextual information and descriptions.
-   **Simulated Multimodal Processing:** For document uploads, a placeholder or simulated output for tasks like OCR, table extraction, and key information extraction will be provided, as actual LLM integration is beyond the scope of a simple specification from a notebook.

## 3. Additional Requirements

### Annotation and Tooltip Specifications

-   **Model Parameters:** When hovering over model names in tables or bars in the parameter chart, a tooltip should display a brief summary of the model (e.g., "Gemma3-4B-IT: A 4 billion parameter instruction-tuned model with a vision encoder.").
-   **Quantization Strategies:** Hovering over quantization strategy labels (e.g., "bfloat16", "Int4", "SFP8", "KV Caching (Yes/No)") should display their respective descriptions, utilizing the `describe_quantization_strategy` function's output.
-   **Chart Elements:** Data points, bars, and lines in all visualizations should provide tooltips showing precise numerical values and relevant labels upon hover.

### Save the states of the fields properly so that changes are not lost

-   The application must leverage `st.session_state` to persist user selections for:
    -   Selected Gemma models.
    -   Uploaded financial documents.
    -   Chosen quantization strategies (type and KV caching).
    -   Selected performance benchmarks for display.
    -   Any other user-configurable parameters to ensure a seamless user experience across reruns or interactions.

## 4. Notebook Content and Code Requirements

This section extracts the relevant markdown and code stubs directly from the Jupyter Notebook and specifies how they will be integrated into the Streamlit application.

### Extracted Markdown Content

The following markdown sections will be rendered in the Streamlit application using `st.markdown()` to provide context and explanations.

-   **Introduction: GemmaVision-QuantAdvisor**
    ```markdown
    ### Introduction: GemmaVision-QuantAdvisor

    Welcome to the GemmaVision-QuantAdvisor Jupyter Notebook! This platform is designed specifically for **Financial Data Engineers** to explore, evaluate, and compare the latest Gemma 3 models, focusing on their capabilities for multimodal financial document understanding, quantization strategies, and performance benchmarks.

    Large Language Models (LLMs) are becoming increasingly important in financial data processing, from automated report analysis to intelligent invoice parsing. Gemma 3, with its enhanced multimodal features and improved efficiency, offers a compelling solution. This notebook aims to provide the necessary insights to make informed deployment decisions tailored to specific hardware constraints and operational costs.

    #### Learning Goals:
    Upon completing this notebook, you will be able to:
    -   Understand the architectural and performance characteristics of Gemma 3 models.
    -   Evaluate the impact of different quantization techniques on memory footprint and efficiency.
    -   Compare Gemma 3's performance in multimodal tasks, general intelligence, math, and reasoning against previous Gemma versions and other state-of-the-art models.
    -   Utilize quantitative benchmarks and visualizations to guide strategic model deployment in financial data engineering workflows.
    ```

-   **Setting Up the Environment**
    ```markdown
    ### Setting Up the Environment

    To ensure the smooth execution of this notebook, we will import the necessary Python libraries. These libraries provide functionalities for data manipulation, numerical operations, and advanced data visualization.
    ```
    ```markdown
    The essential libraries have been imported. `pandas` will be used for structured data, `matplotlib.pyplot` and `seaborn` for visualizations, `numpy` for numerical operations, `Pillow` for simulated image processing, and `math` for utility functions.
    ```

-   **Gemma 3 Model Overview: Parameter Counts**
    ```markdown
    ### Gemma 3 Model Overview: Parameter Counts

    Understanding the scale of a model is crucial for assessing its computational requirements. The Gemma 3 family offers models ranging from 1 to 27 billion parameters, each with specific components like vision encoders, embedding layers, and non-embedding parameters. These counts directly influence the model's complexity and potential performance.

    As per [2, Table 1], the parameter counts for Gemma 3 models are structured as follows:
    -   **Vision Encoder Parameters:** Parameters specific to the vision component.
    -   **Embedding Parameters:** Parameters for token embeddings.
    -   **Non-embedding Parameters:** The majority of the model's parameters, including transformer layers.

    Let $P_V$ be the Vision Encoder Parameters, $P_E$ be the Embedding Parameters, and $P_{NE}$ be the Non-embedding Parameters. The total parameters $P_T$ for a model are given by:
    $$ P_T = P_V + P_E + P_{NE} $$
    All parameter counts are typically expressed in millions.
    ```
    ```markdown
    The parameter counts for the Gemma 3 models have been retrieved and aggregated into a DataFrame. We can observe the distinct parameter distribution across different model sizes, with the 1B model notably lacking a Vision Encoder component, while larger models share the same Vision Encoder but scale up significantly in Embedding and Non-embedding parameters. This data is foundational for understanding the computational complexity of each model.
    ```

-   **Visualizing Model Parameter Counts**
    ```markdown
    ### Visualizing Model Parameter Counts

    A visual representation of the parameter counts provides an immediate understanding of the relative size and complexity of each Gemma 3 model. This is especially useful for Financial Data Engineers when considering the hardware capacity required for deployment.
    ```
    ```markdown
    The stacked bar chart clearly illustrates how the total parameter count scales with model size, and how the distribution across vision encoder, embedding, and non-embedding components changes (or remains constant for the vision encoder across 4B, 12B, 27B models). This visualization highlights the architectural similarities and scaling differences, informing resource allocation for different model sizes.
    ```

-   **Understanding Quantization Strategies**
    ```markdown
    ### Understanding Quantization Strategies

    Quantization is a critical technique for optimizing LLMs for deployment, especially in resource-constrained environments or for reducing operational costs. It involves reducing the precision of model weights and activations, leading to smaller memory footprints and faster inference. Gemma 3 models support various quantization strategies.

    Key quantization concepts include:
    -   **bfloat16 (Brain Float 16):** A 16-bit floating-point format that offers a good balance between range and precision, commonly used for training and raw model checkpoints.
    -   **Int4 (4-bit Integer):** A quantization strategy that represents weights as 4-bit integers, significantly reducing memory usage compared to bfloat16. This often comes with a slight trade-off in accuracy.
    -   **SFP8 (Scaled Float 8):** A less common but emerging 8-bit floating-point format designed for efficiency.
    -   **KV Caching:** Key-Value caching stores intermediate activations from previous tokens to avoid recomputing them, which is essential for long-context inference but consumes significant memory. Quantizing KV cache also helps reduce memory.

    As shown in [3, Table 3], quantization can lead to substantial memory savings for both model weights and KV caching.
    ```

### Extracted Code Stubs and Usage in Streamlit

The following code stubs will be directly integrated into the Streamlit application's Python script (`streamlit_app.py`).

-   **Library Imports:**
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from PIL import Image # For simulated image handling
    import math # For mathematical constants, e.g., for radar charts
    # Streamlit specific import
    import streamlit as st
    ```
    *   **Usage in Streamlit:** These imports will be placed at the very beginning of the `streamlit_app.py` file to ensure all necessary libraries are available.

-   **Gemma Model Parameters Data and Retrieval Function:**
    ```python
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
    ```
    *   **Usage in Streamlit:** This dictionary and function will be used to populate the model selection dropdowns/multiselects and to retrieve data for parameter tables and charts based on user model choices.

-   **Gemma Model Parameters DataFrame Creation Logic:**
    ```python
    # This logic will be triggered based on user selections in Streamlit
    # Example for selected_gemma_models:
    # selected_gemma_models = st.session_state.get('selected_models', ["Gemma3-4B-IT"]) # Default or user selected
    #
    # model_parameters = []
    # for model in selected_gemma_models:
    #     params = get_gemma_model_parameters(model)
    #     if params:
    #         params['Model'] = model
    #         model_parameters.append(params)
    #
    # model_parameters_df = pd.DataFrame(model_parameters)
    # if not model_parameters_df.empty:
    #     model_parameters_df['Total Parameters (M)'] = model_parameters_df[[
    #         "Vision Encoder Parameters", "Embedding Parameters", "Non-embedding Parameters"
    #     ]].sum(axis=1)
    ```
    *   **Usage in Streamlit:** This code block will be dynamically executed within the Streamlit app to generate the `model_parameters_df` based on the models selected by the user via input widgets. The resulting DataFrame will then be displayed using `st.dataframe()` and used for plotting.

-   **Bar Chart Plotting Function:**
    ```python
    def plot_bar_chart(df: pd.DataFrame, x_col: str, y_cols: list, title: str, x_label: str, y_label: str):
        """
        Generates a stacked bar chart of model parameter counts.
        """
        fig, ax = plt.subplots(figsize=(10, 6)) # Create a figure and an axes object
        df.set_index(x_col)[y_cols].plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.xticks(rotation=45, ha='right')
        ax.legend(title='Parameter Type')
        plt.tight_layout()
        # In Streamlit, instead of plt.show(), use st.pyplot()
        st.pyplot(fig) # Pass the figure object to Streamlit
        plt.close(fig) # Close the figure to free up memory
    ```
    *   **Usage in Streamlit:** This function will be called directly in the Streamlit application to render the stacked bar chart of model parameters. The `plt.show()` call from the notebook is replaced with `st.pyplot(fig)` for proper rendering in Streamlit, and `plt.close(fig)` is added for resource management.

-   **Quantization Strategy Description Function:**
    ```python
    def describe_quantization_strategy(strategy_name: str) -> str:
        """
        Returns a descriptive string for a given quantization strategy.
        """
        descriptions = {
            "bfloat16": "Standard 16-bit floating-point precision, often used for raw model weights.",
            "Int4": "4-bit integer quantization, drastically reduces memory but may impact precision.",
            "SFP8": "8-bit scaled floating-point format, offering a balance between bfloat16 and Int4.",
            "KV Cache Status (No)": "Memory used for model weights, with KV cache memory *not* included.",
            "KV Cache Status (Yes)": "Memory used for model weights *and* the Key-Value cache."
        }
        return descriptions.get(strategy_name, "Unknown quantization strategy.")
    ```
    *   **Usage in Streamlit:** This function will be utilized to provide dynamic descriptions, annotations, or tooltips for the quantization options presented to the user. For example, `st.info(describe_quantization_strategy(selected_strategy))` could display the description below the selection widget.
