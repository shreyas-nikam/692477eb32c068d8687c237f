import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_page2():
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

    memory_footprint_data = {
        "Model": ["1B", "1B +KV", "4B", "4B +KV", "12B", "12B +KV", "27B", "27B +KV"],
        "bf16": [2.0, 2.9, 8.0, 12.7, 24.0, 38.9, 54.0, 72.7],
        "Int4": [0.5, 1.4, 2.6, 7.3, 6.6, 21.5, 14.1, 32.8],
        "SFP8": [1.0, 1.9, 4.4, 9.1, 12.4, 27.3, 27.4, 46.1]
    }
    memory_footprint_df = pd.DataFrame(memory_footprint_data)

    st.dataframe(memory_footprint_df.set_index('Model'))

    st.markdown("---")
    st.subheader("Interactive Memory Footprint Visualization")

    weights_only_df = memory_footprint_df[~memory_footprint_df['Model'].str.contains(r'\+KV')]
    kv_cache_df = memory_footprint_df[memory_footprint_df['Model'].str.contains(r'\+KV')].copy()
    kv_cache_df['Model'] = kv_cache_df['Model'].str.replace(r' \+KV', '', regex=True)

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