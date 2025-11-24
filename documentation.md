id: 692477eb32c068d8687c237f_documentation
summary: Gemma 3 Technical Report Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# GemmaVision-QuantAdvisor: A Guide for Financial Data Engineers

## 1. Introduction to GemmaVision-QuantAdvisor and Application Setup
Duration: 00:07:00

Welcome to the **GemmaVision-QuantAdvisor** codelab! This application is designed as an interactive platform for **Financial Data Engineers** to delve into the capabilities of **Gemma 3 models**. It provides a comprehensive guide to understanding their multimodal strengths in financial document processing, evaluating various quantization strategies, and analyzing performance benchmarks.

The application's core purpose is to equip developers and data engineers with the necessary insights to make informed decisions regarding the deployment of Gemma 3 models. By the end of this codelab, you will:

*   **Understand the architectural characteristics** and scaling of Gemma 3 models.
*   **Evaluate the practical impact of different quantization techniques** on memory footprint and computational efficiency, crucial for cost-effective deployment.
*   **Compare Gemma 3's performance** in complex multimodal tasks, general intelligence, mathematical reasoning, and more, against previous Gemma versions and other state-of-the-art models.
*   **Utilize quantitative benchmarks and intuitive visualizations** to strategically integrate these models into financial data engineering workflows.

<aside class="positive">
This application is a blueprint designed to simulate complex interactions and provide data insights. While it offers a deep dive into model characteristics and benchmarks, actual model inference for multimodal tasks is **simulated** to demonstrate potential functionalities without requiring significant computational resources for live model execution.
</aside>

### Application Architecture

The `GemmaVision-QuantAdvisor` is a Streamlit application, which simplifies the creation of interactive web apps using pure Python. Its modular design separates the main application logic from page-specific content, making it easy to navigate and extend.

Here's a high-level overview of the application's structure:

```
gemma_vision_quant_advisor/
├── app.py                      # Main Streamlit application entry point
└── application_pages/
    ├── __init__.py             # Makes application_pages a Python package
    ├── page1.py                # Gemma 3 Model Overview
    ├── page2.py                # Quantization Strategies & Memory Footprint
    ├── page3.py                # Multimodal Document Understanding
    └── page4.py                # Performance Benchmarks
```

The `app.py` file serves as the orchestrator, handling global settings, sidebar navigation, and dynamically loading content from the `application_pages` module based on user selection.

A simple flowchart illustrating the navigation flow:

```
+--+
|       User Starts App    |
|   (streamlit run app.py) |
+--+
          |
          v
+--+
|      app.py (Main UI)    |
|   - Sets page config     |
|   - Displays title       |
|   - Sidebar Navigation   |
+--+
          |
          | Selects Page
          v
+--+
|  "Gemma 3 Model Overview"|
|    (application_pages/   |
|         page1.py)        |
+--+
          |
          v
+--+
| "Quantization Strategies |
|    & Memory Footprint"   |
|    (application_pages/   |
|         page2.py)        |
+--+
          |
          v
+--+
|  "Multimodal Document    |
|      Understanding"      |
|    (application_pages/   |
|         page3.py)        |
+--+
          |
          v
+--+
|   "Performance           |
|      Benchmarks"         |
|    (application_pages/   |
|         page4.py)        |
+--+
```

### Setting Up Your Environment

To run this application, you'll need Python installed (3.8+ recommended) and the necessary libraries.

1.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows, use `.venv\Scripts\activate`
    ```
2.  **Install the required libraries:**
    ```bash
    pip install streamlit pandas matplotlib seaborn numpy Pillow plotly
    ```
3.  **Save the application files:**
    Ensure you have `app.py` in your root directory and the `application_pages` folder containing `page1.py`, `page2.py`, `page3.py`, and `page4.py` in the same directory.
    
    <button>
      [Download Application Files](https://github.com/placeholder_link_to_repo/gemma_vision_quant_advisor)
    </button>
    
4.  **Run the Streamlit application:**
    Navigate to the directory containing `app.py` in your terminal and execute:
    ```bash
    streamlit run app.py
    ```
    Your browser should automatically open to the Streamlit application.

<aside class="positive">
The application handles the imports and initial setup as seen in `app.py`. Upon launching, you will see a confirmation message for successful library imports.
</aside>

Here's the main `app.py` code snippet to give you a clearer picture:

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import math
import plotly.graph_objects as go

st.set_page_config(page_title="GemmaVision-QuantAdvisor", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("GemmaVision-QuantAdvisor")
st.divider()

st.markdown("""
In this lab, the GemmaVision-QuantAdvisor Streamlit application serves as an interactive platform for **Financial Data Engineers** to explore, evaluate, and compare Gemma 3 models. It highlights their multimodal capabilities for financial document understanding, different quantization strategies, and performance benchmarks to aid in informed deployment decisions tailored to specific hardware constraints and operational costs.
...
""")

st.markdown("### Setting Up the Environment")
st.markdown("To ensure the smooth execution of this application, we will import the necessary Python libraries. These libraries provide functionalities for data manipulation, numerical operations, and advanced data visualization.")
st.success("Required libraries imported successfully.")
st.markdown("""
The essential libraries have been imported. `pandas` will be used for structured data, `matplotlib.pyplot` and `seaborn` for visualizations, `numpy` for numerical operations, `Pillow` for simulated image processing, and `math` for utility functions. `streamlit` is for the application framework and `plotly` for advanced visualizations.
""")


page = st.sidebar.selectbox(label="Navigation", options=["Gemma 3 Model Overview", "Quantization Strategies & Memory Footprint", "Multimodal Document Understanding", "Performance Benchmarks"])

if page == "Gemma 3 Model Overview":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "Quantization Strategies & Memory Footprint":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Multimodal Document Understanding":
    from application_pages.page3 import run_page3
    run_page3()
elif page == "Performance Benchmarks":
    from application_pages.page4 import run_page4
    run_page4()
```

## 2. Exploring Gemma 3 Model Overview: Parameter Counts
Duration: 00:10:00

The first page of the application, "Gemma 3 Model Overview," focuses on the scale and architecture of different Gemma 3 models by examining their parameter counts. Understanding model parameters is fundamental for estimating computational requirements, memory footprint, and potential performance.

### Understanding Model Parameter Components

Gemma 3 models are characterized by several parameter types, each contributing to their overall complexity and capabilities:

*   **Vision Encoder Parameters ($P_V$):** Parameters specific to the vision component, crucial for multimodal models like Gemma 3 which can process visual data.
*   **Embedding Parameters ($P_E$):** Parameters used for converting input tokens (text or visual features) into dense vector representations.
*   **Non-embedding Parameters ($P_{NE}$):** The bulk of the model's parameters, primarily residing within the transformer layers responsible for processing and generating outputs.

The total number of parameters ($P_T$) for any given model is the sum of these components:
$$ P_T = P_V + P_E + P_{NE} $$
All parameter counts are typically expressed in millions (M).

The application uses a predefined dictionary `_GEMMA_MODEL_PARAMETERS_DATA` to store this information, simulating data from a technical report.

```python
# From application_pages/page1.py
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
```

You can observe that the `Gemma3-1B` model lacks a Vision Encoder component, indicating it's a text-only model in this context, while larger models like `Gemma3-4B-IT`, `Gemma3-12B-IT`, and `Gemma3-27B-IT` share the same Vision Encoder, scaling primarily in their Embedding and Non-embedding parameters.

### Visualizing Model Parameter Counts

The application provides an interactive stacked bar chart to visualize the distribution of parameter counts across selected Gemma 3 models. This visual aid is invaluable for quickly comparing model sizes and understanding their architectural breakdown.

The `plot_bar_chart` function uses `matplotlib.pyplot` to render the visualization:

```python
# From application_pages/page1.py
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

# Streamlit UI for selection and plotting
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
```

Interact with the `multiselect` widget to compare different Gemma 3 models. Notice how the bar chart dynamically updates, showing the parameter distribution for your chosen models. This helps in understanding which components contribute most to a model's total size.

## 3. Understanding Quantization Strategies & Memory Footprint
Duration: 00:15:00

The "Quantization Strategies & Memory Footprint" page provides critical insights into optimizing LLMs for deployment. Quantization is a technique that reduces the precision of model weights and activations, significantly impacting memory usage and inference speed. This is particularly important for Financial Data Engineers operating under hardware constraints or aiming to reduce operational costs.

### Key Quantization Concepts

*   **bfloat16 (Brain Float 16):** A 16-bit floating-point format that offers a good balance between range and precision. It's often used for training and as the raw checkpoint format for LLMs. A `bfloat16` number is represented with $16$ bits, typically comprising $1$ sign bit, $8$ exponent bits, and $7$ mantissa bits.
*   **Int4 (4-bit Integer):** A highly aggressive quantization strategy that represents weights as $4$-bit integers. This drastically reduces the memory footprint (often by 4x compared to `bfloat16`) but can introduce a slight trade-off in model accuracy.
*   **SFP8 (Scaled Float 8):** An 8-bit floating-point format designed for efficiency. It aims to strike a balance between the memory savings of `Int4` and the precision retention of `bfloat16`.
*   **KV Caching (Key-Value Caching):** During inference, especially for long contexts, the Key and Value states (activations) of previous tokens are stored to avoid recomputing them. This significantly speeds up sequence generation but can consume substantial memory. Quantizing the KV cache further reduces this memory overhead.

The application includes a `describe_quantization_strategy` function to provide explanations for these strategies:

```python
# From application_pages/page2.py
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
```

### Memory Footprints Comparison

The application presents a simulated comparison of memory footprints (in GB) for raw (`bfloat16`) and quantized checkpoints, both for model weights alone and with the inclusion of KV caching (+KV) at a context size of 32,768.

```python
# From application_pages/page2.py
memory_footprint_data = {
    "Model": ["1B", "1B +KV", "4B", "4B +KV", "12B", "12B +KV", "27B", "27B +KV"],
    "bf16": [2.0, 2.9, 8.0, 12.7, 24.0, 38.9, 54.0, 72.7],
    "Int4": [0.5, 1.4, 2.6, 7.3, 6.6, 21.5, 14.1, 32.8],
    "SFP8": [1.0, 1.9, 4.4, 9.1, 12.4, 27.3, 27.4, 46.1]
}
memory_footprint_df = pd.DataFrame(memory_footprint_data)

st.dataframe(memory_footprint_df.set_index('Model'))
```

This table immediately shows the significant memory savings achieved through quantization, especially when considering the large memory demands of KV caching.

### Interactive Memory Footprint Visualization

To make this data more digestible, the application offers an interactive visualization. You can select specific Gemma 3 models, choose a quantization strategy, and toggle whether to include KV cache memory in the comparison.

The data is first preprocessed to separate weights-only data from KV cache data for better plotting control:

```python
# From application_pages/page2.py
weights_only_df = memory_footprint_df[~memory_footprint_df['Model'].str.contains(r'\+KV')]
kv_cache_df = memory_footprint_df[memory_footprint_df['Model'].str.contains(r'\+KV')].copy()
kv_cache_df['Model'] = kv_cache_df['Model'].str.replace(r' \+KV', '', regex=True)

# Streamlit UI for interaction and plotting
st.subheader("Interactive Memory Footprint Visualization")

if 'models_for_memory' not in st.session_state:
    st.session_state.models_for_memory = ["4B", "27B"]

models_for_memory = st.multiselect(
    "Select models to compare memory footprints:",
    options=["1B", "4B", "12B", "27B"],
    default=st.session_state.models_for_memory,
    key="page2_model_selection"
)
st.session_state.models_for_memory = models_for_memory

if models_for_memory:
    col1, col2 = st.columns(2)
    with col1:
        if 'selected_quant_strategy' not in st.session_state:
            st.session_state.selected_quant_strategy = "bf16"
        selected_quant_strategy = st.radio(
            "Select Quantization Strategy:",
            options=["bf16", "Int4", "SFP8"],
            index=["bf16", "Int4", "SFP8"].index(st.session_state.selected_quant_strategy),
            key="mem_quant_strategy"
        )
        st.session_state.selected_quant_strategy = selected_quant_strategy
    with col2:
        if 'show_kv_cache' not in st.session_state:
            st.session_state.show_kv_cache = True
        show_kv_cache = st.checkbox("Show KV Cache Memory", value=st.session_state.show_kv_cache, key="mem_kv_cache")
        st.session_state.show_kv_cache = show_kv_cache

    plot_df = pd.DataFrame()
    for model in models_for_memory:
        weights_data = weights_only_df[weights_only_df['Model'] == model][selected_quant_strategy].iloc[0]
        plot_df = pd.concat([plot_df, pd.DataFrame([{'Model': f"{model} (Weights)", 'Memory (GB)': weights_data}])], ignore_index=True)
        
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

Experiment with the `multiselect`, `radio` buttons, and `checkbox` to observe how different quantization strategies and the inclusion of KV caching impact the memory requirements for various Gemma 3 models. This hands-on visualization directly informs hardware provisioning and deployment strategies.

## 4. Simulating Multimodal Document Understanding
Duration: 00:10:00

The "Multimodal Document Understanding" section allows Financial Data Engineers to explore the simulated capabilities of Gemma 3 models in processing financial documents that combine text and visual information. This is a critical area for automating tasks like invoice processing, annual report analysis, and chart interpretation.

<aside class="negative">
It's important to remember that the actual Gemma 3 model inference for document understanding is **simulated** within this application. A full, live model integration would require significant computational resources and API access, which is beyond the scope of this blueprint. The simulation demonstrates the *potential* outcomes and functionalities.
</aside>

### Document Upload and Analysis Settings

The page starts with a `st.file_uploader` to simulate uploading financial documents. It supports common image and document formats.

```python
# From application_pages/page3.py
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
uploaded_file = st.file_uploader("Upload a Financial Document (JPG, PNG, PDF)", type=["jpg", "png", "pdf"], key="file_uploader")
st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file is not None:
    st.subheader("Uploaded Document")
    if st.session_state.uploaded_file.type == "application/pdf":
        st.warning("PDF processing is conceptual; displaying placeholder for PDF content.")
        st.write("Displaying first page of PDF as image (simulated).")
        st.image("https://via.placeholder.com/600x400.png?text=PDF+Content+Placeholder", caption="Simulated PDF Content")
    else:
        st.image(st.session_state.uploaded_file, caption="Uploaded Image Document", use_column_width=True)

    st.subheader("Multimodal Analysis Settings")
    col1, col2 = st.columns(2)
    with col1:
        if 'multimodal_task' not in st.session_state:
            st.session_state.multimodal_task = "Extract Key Figures (e.g., total amount, date)"
        multimodal_task = st.selectbox(
            "Select Multimodal Task:",
            options=["Extract Key Figures (e.g., total amount, date)", "OCR Text Extraction", "Table Data Extraction"],
            index=["Extract Key Figures (e.g., total amount, date)", "OCR Text Extraction", "Table Data Extraction"].index(st.session_state.multimodal_task),
            key="multimodal_task_selection"
        )
        st.session_state.multimodal_task = multimodal_task
    with col2:
        if 'analysis_model' not in st.session_state:
            st.session_state.analysis_model = "Gemma3-4B-IT"
        analysis_model = st.selectbox(
            "Select Gemma 3 Model for Analysis:",
            options=["Gemma3-4B-IT", "Gemma3-27B-IT"],
            index=["Gemma3-4B-IT", "Gemma3-27B-IT"].index(st.session_state.analysis_model),
            key="analysis_model_selection"
        )
        st.session_state.analysis_model = analysis_model

    if st.button("Run Multimodal Analysis"):
        # ... (analysis simulation logic)
else:
    st.info("Please upload a financial document to perform multimodal analysis.")
```

Once a file is uploaded, you can select from different multimodal tasks and choose a Gemma 3 model for the *simulated* analysis. The application supports:

*   **Extract Key Figures:** Identifying specific data points like total amounts, dates, or item descriptions.
*   **OCR Text Extraction:** Converting image-based text into machine-readable format.
*   **Table Data Extraction:** Identifying and extracting structured data from tables within documents.

### Simulated Analysis Output

Upon clicking "Run Multimodal Analysis," the application simulates the processing and provides a predefined output based on the selected task. This showcases the type of results a real Gemma 3 model could provide.

```python
# From application_pages/page3.py
if st.button("Run Multimodal Analysis"):
    with st.spinner(f"Running {st.session_state.multimodal_task} with {st.session_state.analysis_model}..."):
        st.info("Simulating analysis...")
        if st.session_state.multimodal_task == "Extract Key Figures (e.g., total amount, date)":
            st.success("Key figures extracted successfully (simulated).")
            st.markdown(f"""
            **Simulated Output for {st.session_state.analysis_model}**:
            -   **Total Amount:** $43.07
            -   **Currency:** CHF
            -   **Date:** 04.04.2024
            -   **Item:** Zürcher Geschnetzeltes + Rösti
            -   **Extracted from:** Uploaded Financial Document
            """)
        elif st.session_state.multimodal_task == "OCR Text Extraction":
            st.success("OCR text extracted successfully (simulated).")
            st.text_area(
                "Extracted Text:",
                "This is simulated OCR text from your uploaded financial document. "
                "Actual extraction would involve an OCR model processing the image or PDF. "
                "Example: 'Total CHF: 88.40', '1x Zürcher Geschnetzeltes + Rösti at CHF 36.50'",
                height=200
            )
        elif st.session_state.multimodal_task == "Table Data Extraction":
            st.success("Table data extracted successfully (simulated).")
            simulated_table_data = pd.DataFrame({
                "Item": ["Zürcher Geschnetzeltes + Rösti", "Preiselbeersauce", "4 dl ZHK Hausbier"],
                "Quantity": [1, 1, 2],
                "Unit Price (CHF)": [36.50, 1.80, 6.80],
                "Total (CHF)": [36.50, 1.80, 13.60]
            })
            st.dataframe(simulated_table_data)
```

This simulation helps visualize the potential of Gemma 3 models in automating document-heavy tasks within finance, such as reconciling invoices, processing expense reports, or extracting data from financial statements.

## 5. Analyzing Performance Benchmarks
Duration: 00:15:00

The "Performance Benchmarks" page is crucial for understanding how Gemma 3 models stack up against other leading models. It presents various benchmarks, offering a quantitative view of their general intelligence, reasoning, and specialized abilities.

### Chatbot Arena Elo Scores

The Chatbot Arena uses a blind side-by-side evaluation by human raters to rank models using the Elo rating system, similar to chess. Higher Elo scores indicate better performance as perceived by human evaluators.

```python
# From application_pages/page4.py
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
```

This table gives Financial Data Engineers context on how Gemma 3 models compare in real-world conversational scenarios against a wide array of competitors.

### Summary of Pre-trained Model Abilities (Radar Chart)

A radar chart is an excellent tool for visualizing the performance of models across multiple dimensions or categories simultaneously. This section uses a radar chart to compare Gemma 2 and Gemma 3 models on general abilities like Vision, Code, Science, Factuality, Reasoning, and Multilingual capabilities.

```python
# From application_pages/page4.py
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

categories = radar_df['Category'].tolist()

fig = go.Figure()

if 'selected_radar_models' not in st.session_state:
    st.session_state.selected_radar_models = ['Gemma 3 4B', 'Gemma 3 27B']

selected_radar_models = st.multiselect(
    "Select models for Radar Chart comparison:",
    options=['Gemma 2 2B', 'Gemma 3 4B', 'Gemma 2 9B', 'Gemma 3 12B', 'Gemma 2 27B', 'Gemma 3 27B'],
    default=st.session_state.selected_radar_models,
    key="radar_chart_selection"
)
st.session_state.selected_radar_models = selected_radar_models

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
```

Use the `multiselect` widget to compare different Gemma 2 and Gemma 3 models. Observe how newer Gemma 3 versions generally exhibit improved performance across most categories, particularly in vision due to their enhanced multimodal architecture.

### Detailed Instruction Fine-tuned (IT) Model Benchmarks

This table provides a granular view of instruction fine-tuned (IT) models across various zero-shot benchmarks, giving a comprehensive understanding of their capabilities in specific domains.

```python
# From application_pages/page4.py
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
```

This table allows Financial Data Engineers to directly compare the performance of different Gemma 3 instruction-tuned models against various Gemini models and previous Gemma 2 versions across key benchmarks relevant to real-world applications.

## 6. Conclusion and Further Exploration
Duration: 00:03:00

Congratulations! You have successfully navigated through the **GemmaVision-QuantAdvisor** Streamlit application, gaining valuable insights into the Gemma 3 model family.

In this codelab, you have:
*   **Explored the architectural scale** of Gemma 3 models by analyzing their parameter counts and distributions.
*   **Understood the impact of quantization strategies** (bfloat16, Int4, SFP8) on memory footprints, including the significant role of KV caching.
*   **Simulated multimodal document understanding tasks**, demonstrating Gemma 3's potential in extracting financial information from various document types.
*   **Analyzed comprehensive performance benchmarks**, comparing Gemma 3 with other state-of-the-art models across general abilities and instruction-tuned tasks.

The GemmaVision-QuantAdvisor application serves as a powerful tool for Financial Data Engineers to evaluate and select the most suitable Gemma 3 model for their specific deployment scenarios, balancing performance, resource constraints, and operational costs.

### Next Steps

To further your understanding and application of Gemma 3 models, consider the following:

1.  **Deep Dive into Quantization:** Research the specific implications of different quantization levels on model accuracy for financial tasks.
2.  **Explore Gemma 3 APIs:** Investigate Google's official documentation and APIs for Gemma 3 to integrate these models into your actual data pipelines.
3.  **Real-world Integration:** Experiment with deploying quantized Gemma 3 models on edge devices or cloud platforms to observe actual performance and memory usage.
4.  **Custom Fine-tuning:** Learn about fine-tuning Gemma 3 models on proprietary financial datasets to enhance their performance on niche tasks relevant to your organization.

Thank you for completing this codelab. We hope this guide empowers you to make informed decisions and leverage the full potential of Gemma 3 models in your financial data engineering endeavors!
