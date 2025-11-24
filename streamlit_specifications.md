
# Streamlit Application Requirements Specification

This document outlines the requirements for developing a Streamlit application based on the provided Jupyter Notebook content and user specifications. It details the interactive components, relevant code snippets, and UI/UX considerations to create an effective platform for Financial Data Engineers.

## 1. Application Overview

The `GemmaVision-QuantAdvisor` Streamlit application will serve as an interactive platform for **Financial Data Engineers** to explore, evaluate, and compare Gemma 3 models. It will highlight their multimodal capabilities for financial document understanding, different quantization strategies, and performance benchmarks to aid in informed deployment decisions tailored to specific hardware constraints and operational costs.

### Learning Goals
Upon using this application, users will be able to:
- Understand the architectural and performance characteristics of Gemma 3 models.
- Evaluate the impact of different quantization techniques on memory footprint and efficiency.
- Compare Gemma 3's performance in multimodal tasks, general intelligence, math, and reasoning against previous Gemma versions and other state-of-the-art models.
- Utilize quantitative benchmarks and visualizations to guide strategic model deployment in financial data engineering workflows.

### General Requirements
- Facilitate evaluation of different Gemma 3 models (e.g., Gemma3-4B-IT, Gemma3-27B-IT).
- Provide comparisons against Gemma 2 and competitive frontier models like Gemini-1.5-Pro.
- Focus on metrics relevant to hardware constraints, operational costs, and task performance.
- Enable users to upload financial documents (scanned annual reports, invoices, charts in image or PDF formats) for multimodal analysis, such as extracting key figures or text from images.
- Generate clear and comparative visualizations for model parameters, memory footprints (raw vs. quantized versions including bfloat16, Int4, SFP8 for weights and KV caching), and performance benchmarks (document understanding, math, reasoning, Chatbot Arena, zero-shot benchmarks).
- Incorporate functionality for multimodal document understanding (e.g., OCR, table extraction, key information extraction from images), potentially simulated.
- Allow application and comparison of different quantization strategies for Gemma models.
- Provide a benchmarking module to run uploaded financial documents through selected models and report performance with clear comparison metrics (simulated).

## 2. User Interface Requirements

### Layout and Navigation Structure
The application will utilize a `st.sidebar` for navigation and model selection, with the main content area displaying interactive components and visualizations. Sections will be structured logically to guide the user through model overview, quantization, multimodal tasks, and performance benchmarks.

**Sidebar Navigation:**
-   **Model Selection:** Dropdown or radio buttons to select Gemma 3 models for comparison (e.g., "Gemma3-1B", "Gemma3-4B-IT", "Gemma3-12B-IT", "Gemma3-27B-IT").
-   **Main Content Sections:**
    -   "Gemma 3 Model Overview"
    -   "Quantization Strategies & Memory Footprint"
    -   "Multimodal Document Understanding"
    -   "Performance Benchmarks"

### Input Widgets and Controls

1.  **Model Selection (Sidebar):**
    -   `st.selectbox` or `st.radio` for selecting models for parameter comparison and memory analysis.
    -   Example: `selected_models = st.multiselect("Select Gemma 3 Models for Comparison", options=gemma_models, default=["Gemma3-4B-IT", "Gemma3-27B-IT"])`

2.  **Quantization Strategy Selection (Main Content - "Quantization Strategies" section):**
    -   `st.radio` or `st.selectbox` for choosing a quantization strategy (e.g., "bfloat16", "Int4", "SFP8").
    -   `st.checkbox` for toggling KV Cache inclusion.
    -   Example: `quant_strategy = st.radio("Select Quantization Strategy", options=["bfloat16", "Int4", "SFP8"], index=0)`
    -   Example: `include_kv_cache = st.checkbox("Include KV Cache Memory", value=False)`

3.  **Document Upload (Main Content - "Multimodal Document Understanding" section):**
    -   `st.file_uploader` for uploading financial documents (PDF, JPG, PNG).
    -   Example: `uploaded_file = st.file_uploader("Upload a Financial Document (PDF, JPG, PNG)", type=["pdf", "jpg", "png"])`

4.  **Multimodal Task Selection (Main Content - "Multimodal Document Understanding" section):**
    -   `st.selectbox` for choosing a task (e.g., "Extract Key Figures", "OCR Text", "Table Extraction").
    -   Example: `multimodal_task = st.selectbox("Select Multimodal Task", options=["Extract Key Figures", "OCR Text", "Table Extraction"])`

5.  **Model for Document Analysis (Main Content - "Multimodal Document Understanding" section):**
    -   `st.selectbox` for selecting a Gemma 3 model to perform the analysis on the uploaded document.
    -   Example: `analysis_model = st.selectbox("Select Model for Document Analysis", options=["Gemma3-4B-IT", "Gemma3-27B-IT"])`

### Visualization Components

1.  **Model Parameter Counts:**
    -   **Table:** `st.dataframe` to display `model_parameters_df`.
    -   **Stacked Bar Chart:** `st.pyplot` or `st.bar_chart` to visualize parameter distribution for selected models.

2.  **Memory Footprint:**
    -   **Table:** `st.dataframe` to show raw data for memory footprints from Table 3 of the technical report.
    -   **Grouped Bar Chart:** `st.pyplot` or `st.bar_chart` to compare memory footprints across models and quantization strategies, with an option to include/exclude KV caching.

3.  **Multimodal Document Understanding Output:**
    -   `st.image` to display the uploaded document.
    -   `st.text_area` or `st.dataframe` to show extracted information (simulated output).

4.  **Performance Benchmarks:**
    -   **Chatbot Arena Elo (Table 5 equivalent):** `st.dataframe` for raw data, `st.bar_chart` for visual comparison of Elo scores.
    -   **Zero-shot Benchmarks (Table 6 equivalent):** `st.dataframe` to display various benchmark scores.
    -   **Radar Charts (Figure 2 equivalent):** `st.plotly_chart` using `plotly.graph_objects.Figure` to visualize pre-trained model abilities across categories (Vision, Code, Science, Factuality, Reasoning, Multilingual) for selected Gemma 2 and Gemma 3 models.

### Interactive Elements and Feedback Mechanisms

-   **Loading Spinners:** `st.spinner("Processing...")` for document uploads and model analyses.
-   **Success/Error Messages:** `st.success("Libraries imported successfully.")`, `st.error("Error during processing.")`.
-   **Dynamic Updates:** Charts and tables should automatically update based on user selections in the sidebar or main content.
-   **Tooltips:** Informative tooltips on charts, input fields, and key terms using `st.help` or custom `st.markdown` with definitions.

## 3. Additional Requirements

### Annotation and Tooltip Specifications
-   **Charts and Tables:** Provide tooltips on chart elements (e.g., bar segments, data points) to show exact values or specific model details.
-   **Input Widgets:** Add descriptive tooltips for `st.file_uploader`, `st.selectbox`, `st.radio` to clarify their purpose.
-   **Key Concepts:** For terms like "quantization," "KV Caching," "bfloat16," "Int4," "SFP8," provide concise explanations in nearby markdown or via tooltips to enhance user understanding.

### Save the States of the Fields Properly
The application must leverage `st.session_state` to maintain the state of all input widgets and selections across reruns. This ensures that user choices (e.g., selected models, uploaded files, chosen quantization strategies) are not lost when new interactions trigger a Streamlit rerun.
-   Example: `if 'selected_models' not in st.session_state: st.session_state.selected_models = ["Gemma3-4B-IT"]`

## 4. Notebook Content and Code Requirements

This section outlines how the provided Jupyter Notebook content will be integrated into the Streamlit application, including extracted code stubs and markdown.

### 4.1. Introduction: GemmaVision-QuantAdvisor

**Streamlit Implementation:**
Display this content using `st.markdown()`.

```python
st.markdown("""
### Introduction: GemmaVision-QuantAdvisor

Welcome to the GemmaVision-QuantAdvisor Streamlit Application! This platform is designed specifically for **Financial Data Engineers** to explore, evaluate, and compare the latest Gemma 3 models, focusing on their capabilities for multimodal financial document understanding, quantization strategies, and performance benchmarks.

Large Language Models (LLMs) are becoming increasingly important in financial data processing, from automated report analysis to intelligent invoice parsing. Gemma 3, with its enhanced multimodal features and improved efficiency, offers a compelling solution. This application aims to provide the necessary insights to make informed deployment decisions tailored to specific hardware constraints and operational costs.

#### Learning Goals:
Upon completing this application, you will be able to:
-   Understand the architectural and performance characteristics of Gemma 3 models.
-   Evaluate the impact of different quantization techniques on memory footprint and efficiency.
-   Compare Gemma 3's performance in multimodal tasks, general intelligence, math, and reasoning against previous Gemma versions and other state-of-the-art models.
-   Utilize quantitative benchmarks and visualizations to guide strategic model deployment in financial data engineering workflows.
""")
```

### 4.2. Setting Up the Environment

**Streamlit Implementation:**
Display the markdown and execute the Python imports at the beginning of the `app.py` script. The print statement can be replaced by `st.success()` or `st.info()`.

```python
st.markdown("### Setting Up the Environment")
st.markdown("To ensure the smooth execution of this application, we will import the necessary Python libraries. These libraries provide functionalities for data manipulation, numerical operations, and advanced data visualization.")

# Code Cell for Imports
# These imports would be at the top of the Streamlit script
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image # For simulated image handling
import math # For mathematical constants, e.g., for radar charts
import streamlit as st
import plotly.graph_objects as go # For advanced charts like radar charts

st.success("Required libraries imported successfully.")

st.markdown("""
The essential libraries have been imported. `pandas` will be used for structured data, `matplotlib.pyplot` and `seaborn` for visualizations, `numpy` for numerical operations, `Pillow` for simulated image processing, and `math` for utility functions. `streamlit` is for the application framework and `plotly` for advanced visualizations.
""")
```

### 4.3. Gemma 3 Model Overview: Parameter Counts

**Streamlit Implementation:**
Display the markdown content, including the LaTeX equation. Use `st.dataframe()` for the `model_parameters_df` and integrate the `plot_bar_chart` function for visualization.

```python
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

# Code Cell: Data and function definition
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

# Code Cell: Data processing and display
gemma_models = list(_GEMMA_MODEL_PARAMETERS_DATA.keys()) # All available models

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

# Code Cell: Plotting function for Streamlit
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

# Code Cell: Plotting execution in Streamlit
st.write("---") # Separator
st.subheader("Gemma 3 Model Parameter Distribution")
selected_models_for_plot = st.multiselect(
    "Select models to visualize parameter counts:",
    options=gemma_models,
    default=["Gemma3-1B", "Gemma3-4B-IT", "Gemma3-12B-IT", "Gemma3-27B-IT"]
)
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
```

### 4.4. Understanding Quantization Strategies

**Streamlit Implementation:**
Display the markdown content with LaTeX for key concepts. Include the `describe_quantization_strategy` function. For memory footprint, create a `pd.DataFrame` based on Table 3 from the PDF and visualize it.

```python
st.markdown("### Understanding Quantization Strategies")
st.markdown("""
Quantization is a critical technique for optimizing LLMs for deployment, especially in resource-constrained environments or for reducing operational costs. It involves reducing the precision of model weights and activations, leading to smaller memory footprints and faster inference. Gemma 3 models support various quantization strategies.

Key quantization concepts include:
-   **bfloat16 (Brain Float 16):** A 16-bit floating-point format that offers a good balance between range and precision, commonly used for training and raw model checkpoints. Represented with $16$ bits, typically $1$ sign bit, $8$ exponent bits, and $7$ mantissa bits.
-   **Int4 (4-bit Integer):** A quantization strategy that represents weights as $4$-bit integers, significantly reducing memory usage compared to bfloat16. This often comes with a slight trade-off in accuracy.
-   **SFP8 (Scaled Float 8):** A less common but emerging $8$-bit floating-point format designed for efficiency.
-   **KV Caching:** Key-Value caching stores intermediate activations from previous tokens to avoid recomputing them, which is essential for long-context inference but consumes significant memory. Quantizing KV cache also helps reduce memory.

As shown in [3, Table 3], quantization can lead to substantial memory savings for both model weights and KV caching.
""")

# Code Cell: describe_quantization_strategy function
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

st.markdown(f"**Description of bfloat16:** {describe_quantization_strategy('bfloat16')}")
st.markdown(f"**Description of Int4:** {describe_quantization_strategy('Int4')}")
st.markdown(f"**Description of SFP8:** {describe_quantization_strategy('SFP8')}")

st.subheader("Memory Footprints (in GB) Comparison")
st.markdown("Below is a comparison of memory footprints for raw (bfloat16) and quantized checkpoints for weights and KV caching (+KV) at 32,768 context size, based on Table 3 from the technical report.")

# Data from Table 3 of the technical report (OCR'ed content)
memory_footprint_data = {
    "Model": ["1B", "1B +KV", "4B", "4B +KV", "12B", "12B +KV", "27B", "27B +KV"],
    "bf16": [2.0, 2.9, 8.0, 12.7, 24.0, 38.9, 54.0, 72.7],
    "Int4": [0.5, 1.4, 2.6, 7.3, 6.6, 21.5, 14.1, 32.8],
    "SFP8": [1.0, 1.9, 4.4, 9.1, 12.4, 27.3, 27.4, 46.1] # Int4blocks=32 is ignored for simplicity here
}
memory_footprint_df = pd.DataFrame(memory_footprint_data)

st.dataframe(memory_footprint_df.set_index('Model'))

st.markdown("---")
st.subheader("Interactive Memory Footprint Visualization")

# Separate data for weights only and weights + KV cache
weights_only_df = memory_footprint_df[~memory_footprint_df['Model'].str.contains(r'\+KV')]
kv_cache_df = memory_footprint_df[memory_footprint_df['Model'].str.contains(r'\+KV')]
kv_cache_df['Model'] = kv_cache_df['Model'].str.replace(r' \+KV', '', regex=True)

models_for_memory = st.multiselect(
    "Select models to compare memory footprints:",
    options=["1B", "4B", "12B", "27B"],
    default=["4B", "27B"]
)

if models_for_memory:
    col1, col2 = st.columns(2)
    with col1:
        selected_quant_strategy = st.radio(
            "Select Quantization Strategy:",
            options=["bf16", "Int4", "SFP8"],
            index=0,
            key="mem_quant_strategy"
        )
    with col2:
        show_kv_cache = st.checkbox("Show KV Cache Memory", value=True, key="mem_kv_cache")

    plot_df = pd.DataFrame()
    for model in models_for_memory:
        # Get weights only
        weights_data = weights_only_df[weights_only_df['Model'] == model][selected_quant_strategy].iloc[0]
        plot_df = pd.concat([plot_df, pd.DataFrame([{'Model': f"{model} (Weights)", 'Memory (GB)': weights_data}])], ignore_index=True)
        
        # Get KV cache if selected
        if show_kv_cache:
            kv_data = kv_cache_df[kv_cache_df['Model'] == model][selected_quant_strategy].iloc[0]
            plot_df = pd.concat([plot_df, pd.DataFrame([{'Model': f"{model} (Weights + KV)", 'Memory (GB)': kv_data}])], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Model', y='Memory (GB)', data=plot_df, palette='viridis', ax=ax)
    ax.set_title(f'Memory Footprint for {selected_quant_strategy} (GB)')
    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('Memory (GB)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Please select at least one model to visualize memory footprints.")

```

### 4.5. Multimodal Document Understanding

**Streamlit Implementation:**
This section will simulate the multimodal capabilities as no direct code for model inference is provided in the notebook for this part. It will use `st.file_uploader`, `st.image`, and `st.text_area` for input/output.

```python
st.markdown("### Multimodal Document Understanding")
st.markdown("""
This section allows Financial Data Engineers to simulate multimodal tasks using Gemma 3 models. You can upload financial documents such as scanned annual reports, invoices, or charts and select a task to extract key information.

**Note:** The actual Gemma 3 model inference for document understanding is simulated for this application, as a full model integration is beyond the scope of this blueprint.
""")

uploaded_file = st.file_uploader("Upload a Financial Document (JPG, PNG, PDF)", type=["jpg", "png", "pdf"])

if uploaded_file is not None:
    st.subheader("Uploaded Document")
    if uploaded_file.type == "application/pdf":
        st.warning("PDF processing is conceptual; displaying placeholder for PDF content.")
        st.write("Displaying first page of PDF as image (simulated).")
        # In a real app, you'd convert PDF to image for display or use a PDF viewer component
        # For this spec, we just acknowledge it.
        st.image("https://via.placeholder.com/600x400.png?text=PDF+Content+Placeholder", caption="Simulated PDF Content")
    else:
        st.image(uploaded_file, caption="Uploaded Image Document", use_column_width=True)

    st.subheader("Multimodal Analysis Settings")
    col1, col2 = st.columns(2)
    with col1:
        multimodal_task = st.selectbox(
            "Select Multimodal Task:",
            options=["Extract Key Figures (e.g., total amount, date)", "OCR Text Extraction", "Table Data Extraction"],
            index=0
        )
    with col2:
        analysis_model = st.selectbox(
            "Select Gemma 3 Model for Analysis:",
            options=["Gemma3-4B-IT", "Gemma3-27B-IT"], # Focus on multimodal-capable models
            index=0
        )

    if st.button("Run Multimodal Analysis"):
        with st.spinner(f"Running {multimodal_task} with {analysis_model}..."):
            st.info("Simulating analysis...")
            # Simulated output based on task and model
            if multimodal_task == "Extract Key Figures (e.g., total amount, date)":
                st.success("Key figures extracted successfully (simulated).")
                st.markdown(f"""
                **Simulated Output for {analysis_model}**:
                -   **Total Amount:** $43.07
                -   **Currency:** CHF
                -   **Date:** 04.04.2024
                -   **Item:** Zürcher Geschnetzeltes + Rösti
                -   **Extracted from:** Uploaded Financial Document
                """)
            elif multimodal_task == "OCR Text Extraction":
                st.success("OCR text extracted successfully (simulated).")
                st.text_area(
                    "Extracted Text:",
                    "This is simulated OCR text from your uploaded financial document. "
                    "Actual extraction would involve an OCR model processing the image or PDF. "
                    "Example: 'Total CHF: 88.40', '1x Zürcher Geschnetzeltes + Rösti at CHF 36.50'",
                    height=200
                )
            elif multimodal_task == "Table Data Extraction":
                st.success("Table data extracted successfully (simulated).")
                simulated_table_data = pd.DataFrame({
                    "Item": ["Zürcher Geschnetzeltes + Rösti", "Preiselbeersauce", "4 dl ZHK Hausbier"],
                    "Quantity": [1, 1, 2],
                    "Unit Price (CHF)": [36.50, 1.80, 6.80],
                    "Total (CHF)": [36.50, 1.80, 13.60]
                })
                st.dataframe(simulated_table_data)
else:
    st.info("Please upload a financial document to perform multimodal analysis.")
```

### 4.6. Performance Benchmarks

**Streamlit Implementation:**
This section will present key performance benchmarks using `st.dataframe` and `st.plotly_chart` for radar charts (Figure 2 equivalent). Data for these will be extracted or synthesized from the provided technical report tables (Table 5, Table 6, etc.).

```python
st.markdown("### Performance Benchmarks")
st.markdown("""
This section provides a comparative overview of Gemma 3 models against other state-of-the-art models on various benchmarks, including Chatbot Arena scores, zero-shot benchmarks for general abilities, and multimodal performance.
""")

st.subheader("Chatbot Arena Elo Scores (Table 5 equivalent)")
st.markdown("""
Evaluation of Gemma 3 27B IT model in the Chatbot Arena based on blind side-by-side evaluations by human raters. Scores are based on the Elo rating system. (Data from Table 5 of the technical report).
""")

# Data from Table 5
chatbot_arena_data = {
    "Rank": [1, 1, 3, 3, 3, 6, 6, 8, 9, 9, 9, 9, 13, 14, 14, 14, 14, 18, 18, 18, 18, 28, 38, 39, 59],
    "Model": ["Grok-3-Preview-02-24", "GPT-4.5-Preview", "Gemini-2.0-Flash-Thinking-Exp-01-21",
              "Gemini-2.0-Pro-Exp-02-05", "ChatGPT-40-latest (2025-01-29)", "DeepSeek-R1",
              "Gemini-2.0-Flash-001", "01-2024-12-17", "Gemma-3-27B-IT", "Qwen2.5-Max",
              "01-preview", "03-mini-high", "DeepSeek-V3", "GLM-4-Plus-0111", "Qwen-Plus-0125",
              "Claude 3.7 Sonnet", "Gemini-2.0-Flash-Lite", "Step-2-16K-Exp", "03-mini",
              "01-mini", "Gemini-1.5-Pro-002", "Meta-Llama-3.1-405B-Instruct-bf16",
              "Llama-3.3-70B-Instruct", "Qwen2.5-72B-Instruct", "Gemma-2-27B-it"],
    "Elo": [1412, 1411, 1384, 1380, 1377, 1363, 1357, 1352, 1338, 1336, 1335, 1329,
            1318, 1311, 1310, 1309, 1308, 1305, 1304, 1304, 1302, 1269, 1257, 1257, 1220]
}
chatbot_arena_df = pd.DataFrame(chatbot_arena_data)
st.dataframe(chatbot_arena_df.set_index('Rank'))

st.markdown("---")
st.subheader("Summary of Pre-trained Model Abilities (Figure 2 equivalent)")
st.markdown("""
This radar chart provides a simplified summary of the performance of different pre-trained models from Gemma 2 and Gemma 3 across general abilities (Vision, Code, Science, Factuality, Reasoning, Multilingual). (Conceptual visualization based on Figure 2 from the technical report).
""")

# Data for Radar Chart (Synthesized based on Figure 2 concept)
radar_data = {
    'Category': ['Vision', 'Code', 'Science', 'Factuality', 'Reasoning', 'Multilingual'],
    'Gemma 2 2B': [0.5, 0.4, 0.6, 0.5, 0.4, 0.3],
    'Gemma 3 4B': [0.7, 0.6, 0.7, 0.6, 0.5, 0.5],
    'Gemma 2 9B': [0.6, 0.5, 0.7, 0.6, 0.5, 0.4],
    'Gemma 3 12B': [0.8, 0.7, 0.8, 0.7, 0.6, 0.6],
    'Gemma 2 27B': [0.7, 0.6, 0.8, 0.7, 0.6, 0.5],
    'Gemma 3 27B': [0.9, 0.8, 0.9, 0.8, 0.7, 0.7]
}
radar_df = pd.DataFrame(radar_data)

# Create Plotly Radar Chart
categories = radar_df['Category'].tolist()

fig = go.Figure()

selected_radar_models = st.multiselect(
    "Select models for Radar Chart comparison:",
    options=['Gemma 2 2B', 'Gemma 3 4B', 'Gemma 2 9B', 'Gemma 3 12B', 'Gemma 2 27B', 'Gemma 3 27B'],
    default=['Gemma 3 4B', 'Gemma 3 27B']
)

for model in selected_radar_models:
    fig.add_trace(go.Scatterpolar(
        r=radar_df[model].tolist(),
        theta=categories,
        fill='toself',
        name=model
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title="Gemma 2 vs. Gemma 3 Pre-trained Model Abilities"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
Overall, the radar charts indicate that newer Gemma 3 versions generally show improved performance across most categories, with a notable enhancement in vision capabilities due to their multimodal architecture.
""")

st.markdown("---")
st.subheader("Detailed Instruction Fine-tuned (IT) Model Benchmarks")
st.markdown("""
Performance of instruction fine-tuned (IT) models compared to Gemini 1.5, Gemini 2.0, and Gemma 2 on zero-shot benchmarks across different abilities (Data from Table 6 of the technical report).
""")

# Data from Table 6
it_benchmarks_data = {
    "Benchmark": ["MMLU-Pro", "LiveCodeBench", "Bird-SQL (dev)", "GPQA Diamond", "SimpleQA", "FACTS Grounding",
                  "Global MMLU-Lite", "MATH", "HiddenMath", "MMMU (val)"],
    "Gemini 1.5 Flash": [67.3, 30.7, 45.6, 51.0, 8.6, 82.9, 73.7, 77.9, 47.2, 62.3],
    "Gemini 1.5 Pro": [75.8, 34.2, 54.4, 59.1, 24.9, 80.0, 80.8, 86.5, 52.0, 65.9],
    "Gemini 2.0 Flash": [77.6, 34.5, 58.7, 60.1, 29.9, 84.6, 83.4, 90.9, 63.5, 71.7],
    "Gemini 2.0 Pro": [79.1, 36.0, 59.3, 64.7, 44.3, 82.8, 86.5, 91.8, 65.2, 72.7],
    "Gemma 2 2B": [15.6, 1.2, 12.2, 24.7, 2.8, 43.8, 41.9, 27.2, 1.8, None],
    "Gemma 2 9B": [46.8, 10.8, 33.8, 28.8, 5.3, 62.0, 64.8, 49.4, 10.4, None],
    "Gemma 2 27B": [56.9, 20.4, 46.7, 34.3, 9.2, 62.4, 68.6, 55.6, 14.8, None],
    "Gemma 3 1B": [14.7, 1.9, 6.4, 19.2, 2.2, 36.4, 34.2, 48.0, 15.8, None],
    "Gemma 3 4B": [43.6, 12.6, 36.3, 30.8, 4.0, 70.1, 54.5, 75.6, 43.0, 48.8],
    "Gemma 3 12B": [60.6, 24.6, 47.9, 40.9, 6.3, 75.8, 69.5, 83.8, 54.5, 59.6],
    "Gemma 3 27B": [67.5, 29.7, 54.4, 42.4, 10.0, 74.9, 75.1, 89.0, 60.3, 64.9]
}
it_benchmarks_df = pd.DataFrame(it_benchmarks_data)
st.dataframe(it_benchmarks_df.set_index('Benchmark'))

st.markdown("""
The detailed table shows the performance of various instruction-tuned models across a range of benchmarks, providing a comprehensive view of their capabilities in different domains.
""")
```
