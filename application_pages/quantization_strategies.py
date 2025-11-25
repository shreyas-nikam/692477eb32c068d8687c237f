
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data(ttl="2h")
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

# Dummy data for memory footprint comparison based on the specification
# "Table 3 | Memory footprints (in GB) comparison between raw (bfloat16) and quantized checkpoints for weights and KV caching (+KV) at 32,768 context size, quantized in 8 bits"
_MEMORY_FOOTPRINT_DATA = {
    "Model": ["Gemma3-1B", "Gemma3-1B", "Gemma3-1B", "Gemma3-1B",
              "Gemma3-4B-IT", "Gemma3-4B-IT", "Gemma3-4B-IT", "Gemma3-4B-IT",
              "Gemma3-12B-IT", "Gemma3-12B-IT", "Gemma3-12B-IT", "Gemma3-12B-IT",
              "Gemma3-27B-IT", "Gemma3-27B-IT", "Gemma3-27B-IT", "Gemma3-27B-IT"],
    "Quantization": ["bfloat16", "bfloat16", "Int4", "Int4",
                     "bfloat16", "bfloat16", "Int4", "Int4",
                     "bfloat16", "bfloat16", "Int4", "Int4",
                     "bfloat16", "bfloat16", "Int4", "Int4"],
    "KV Caching": ["No", "Yes", "No", "Yes",
                   "No", "Yes", "No", "Yes",
                   "No", "Yes", "No", "Yes",
                   "No", "Yes", "No", "Yes"],
    "Memory (GB)": [2.0, 3.5, 0.5, 1.0,
                    8.0, 14.0, 2.0, 3.5,
                    24.0, 42.0, 6.0, 10.5,
                    54.0, 94.5, 13.5, 23.5] # Example values, not from actual report
}

@st.cache_data(ttl="2h")
def get_memory_footprint_df() -> pd.DataFrame:
    return pd.DataFrame(_MEMORY_FOOTPRINT_DATA)

def plot_memory_footprint_chart(df: pd.DataFrame, title: str):
    """
    Generates a grouped bar chart for memory footprint comparison.
    """
    fig = plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x='Model', y='Memory (GB)', hue='Quantization', 
                errorbar=None, palette='deep', hue_order=['bfloat16', 'Int4'])
    plt.title(title)
    plt.xlabel('Gemma 3 Model')
    plt.ylabel('Memory Footprint (GB)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Quantization Strategy')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def main():
    st.markdown("""
    ### Understanding Quantization Strategies

    Quantization is a critical technique for optimizing LLMs for deployment, especially in resource-constrained environments or for reducing operational costs. It involves reducing the precision of model weights and activations, leading to smaller memory footprints and faster inference. Gemma 3 models support various quantization strategies.

    Key quantization concepts include:
    -   **bfloat16 (Brain Float 16):** A 16-bit floating-point format that offers a good balance between range and precision, commonly used for training and raw model checkpoints.
    -   **Int4 (4-bit Integer):** A quantization strategy that represents weights as 4-bit integers, significantly reducing memory usage compared to bfloat16. This often comes with a slight trade-off in accuracy.
    -   **SFP8 (Scaled Float 8):** A less common but emerging 8-bit floating-point format designed for efficiency.
    -   **KV Caching:** Key-Value caching stores intermediate activations from previous tokens to avoid recomputing them, which is essential for long-context inference but consumes significant memory. Quantizing KV cache also helps reduce memory.

    As shown in [3, Table 3], quantization can lead to substantial memory savings for both model weights and KV caching.
    """)

    st.markdown("#### Quantization Strategy Selection")
    
    quantization_options = ["bfloat16", "Int4", "SFP8"]
    selected_quantization_type = st.radio(
        "Select Quantization Type:",
        options=quantization_options,
        index=quantization_options.index("Int4"), # Default to Int4
        key="selected_quantization_type"
    )
    st.info(describe_quantization_strategy(selected_quantization_type))

    kv_caching_enabled = st.checkbox(
        "Enable KV Caching?",
        value=True, # Default to Yes
        key="kv_caching_enabled"
    )
    kv_caching_status = "KV Cache Status (Yes)" if kv_caching_enabled else "KV Cache Status (No)"
    st.info(describe_quantization_strategy(kv_caching_status))

    st.markdown("#### Memory Footprint Comparison")
    memory_df = get_memory_footprint_df()

    # Filter based on selected KV Caching status
    filtered_memory_df = memory_df[memory_df["KV Caching"] == ("Yes" if kv_caching_enabled else "No")]
    
    # Further filter based on selected quantization type for display table
    display_memory_df = filtered_memory_df[filtered_memory_df["Quantization"].isin([selected_quantization_type, "bfloat16"])]
    
    st.dataframe(display_memory_df)

    st.markdown("##### Memory Usage Across Models and Quantization Strategies")
    plot_memory_footprint_chart(
        filtered_memory_df,
        title=f"Memory Footprint Comparison with KV Caching: {kv_caching_enabled}"
    )
