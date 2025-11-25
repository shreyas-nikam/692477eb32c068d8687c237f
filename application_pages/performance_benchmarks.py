
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math # For radar chart if implemented

# Dummy Data for Chatbot Arena Elo Scores
_CHATBOT_ARENA_ELO_DATA = {
    "Model": ["Gemma 3 27B IT", "GPT-4", "Claude 3 Opus", "Gemini 1.5 Pro", "Gemma 2 27B IT"],
    "Elo Score": [1250, 1300, 1280, 1220, 1100]
}

@st.cache_data(ttl="2h")
def get_chatbot_arena_elo_scores_df() -> pd.DataFrame:
    return pd.DataFrame(_CHATBOT_ARENA_ELO_DATA)

def plot_elo_scores_bar_chart(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='Model', y='Elo Score', ax=ax, palette='viridis')
    ax.set_title(title)
    ax.set_xlabel('Model')
    ax.set_ylabel('Elo Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Dummy Data for Zero-shot General Abilities (Table 6 and Figure 2 inspired)
_ZERO_SHOT_ABILITIES_DATA = {
    "Model": ["Gemma 3 27B IT", "Gemini 1.5 Pro", "Gemma 2 27B", "Gemma 3 4B IT"],
    "Vision": [75, 80, 60, 50],
    "Code": [70, 75, 55, 45],
    "Science": [80, 85, 65, 55],
    "Factuality": [78, 82, 62, 52],
    "Reasoning": [72, 78, 58, 48],
    "Multilingual": [65, 70, 50, 40]
}

@st.cache_data(ttl="2h")
def get_zero_shot_abilities_df() -> pd.DataFrame:
    return pd.DataFrame(_ZERO_SHOT_ABILITIES_DATA)

def plot_radar_chart(df: pd.DataFrame, models: list, categories: list, title: str):
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Complete the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model_name in models:
        values = df[df['Model'] == model_name][categories].iloc[0].tolist()
        values += values[:1] # Complete the loop for plotting
        ax.plot(angles, values, label=model_name)
        ax.fill(angles, values, alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(title, va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# Dummy Data for Other Benchmark Tables (generic example)
_OTHER_BENCHMARKS_DATA = {
    "Benchmark": ["Financial QA", "Legal Document Summarization", "Sentiment Analysis (Financial News)", "Code Generation"],
    "Gemma 3 27B IT": [85, 78, 92, 70],
    "Gemma 2 27B IT": [70, 65, 80, 60],
    "Gemini 1.5 Pro": [90, 85, 95, 75]
}

@st.cache_data(ttl="2h")
def get_other_benchmarks_df() -> pd.DataFrame:
    return pd.DataFrame(_OTHER_BENCHMARKS_DATA)

# Dummy Data for Long Context Performance (Figure 7 inspired)
_LONG_CONTEXT_PERFORMANCE_DATA = {
    "Model": ["Gemma 3 27B Pre-trained", "Gemma 3 27B IT (RoPE Rescaling)", "Gemma 2 27B Pre-trained"],
    "Context Length": [1024, 2048, 4096, 8192, 16384, 32768, 65536],
    "Perplexity_Gemma3_PT": [5.0, 4.8, 4.6, 4.5, 4.4, 4.3, 4.2],
    "Perplexity_Gemma3_IT_RoPE": [4.9, 4.7, 4.5, 4.3, 4.1, 4.0, 3.9],
    "Perplexity_Gemma2_PT": [5.5, 5.3, 5.2, 5.1, 5.0, 4.9, 4.8]
}

@st.cache_data(ttl="2h")
def get_long_context_performance_df() -> pd.DataFrame:
    # Need to reshape this for plotting.
    data = []
    for i, length in enumerate(_LONG_CONTEXT_PERFORMANCE_DATA["Context Length"]):
        data.append({
            "Model": "Gemma 3 27B Pre-trained",
            "Context Length": length,
            "Perplexity": _LONG_CONTEXT_PERFORMANCE_DATA["Perplexity_Gemma3_PT"][i]
        })
        data.append({
            "Model": "Gemma 3 27B IT (RoPE Rescaling)",
            "Context Length": length,
            "Perplexity": _LONG_CONTEXT_PERFORMANCE_DATA["Perplexity_Gemma3_IT_RoPE"][i]
        })
        data.append({
            "Model": "Gemma 2 27B Pre-trained",
            "Context Length": length,
            "Perplexity": _LONG_CONTEXT_PERFORMANCE_DATA["Perplexity_Gemma2_PT"][i]
        })
    return pd.DataFrame(data)


def plot_long_context_line_chart(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x='Context Length', y='Perplexity', hue='Model', marker='o', ax=ax, palette='deep')
    ax.set_title(title)
    ax.set_xlabel('Context Length')
    ax.set_ylabel('Perplexity')
    ax.set_xscale('log', base=2) # Context length often on log scale
    plt.xticks(df['Context Length'].unique(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.markdown("""
    ### Performance Benchmarks

    This section provides a detailed look into the performance benchmarks of Gemma 3 models across various capabilities, including general intelligence, multimodal tasks, math, and reasoning. These benchmarks are crucial for Financial Data Engineers to understand the real-world applicability and efficiency of these models in different scenarios.
    """)

    st.markdown("#### Select Benchmarks to Display")
    benchmark_options = [
        "Chatbot Arena Elo Scores",
        "Zero-shot General Abilities",
        "Other Benchmarks (e.g., Financial QA)",
        "Long Context Performance"
    ]
    selected_benchmarks = st.multiselect(
        "Choose performance benchmarks:",
        options=benchmark_options,
        default=benchmark_options,
        key="selected_performance_benchmarks"
    )

    if st.button("Generate Visualizations", key="generate_benchmark_viz_button"):
        if "Chatbot Arena Elo Scores" in selected_benchmarks:
            st.markdown("---")
            st.markdown("#### Chatbot Arena Elo Scores")
            st.markdown("""
            Table 5 | Evaluation of Gemma 3 27B IT model in the Chatbot Arena: This table shows the relative ranking of Gemma 3 models against other competitive models based on Elo scores from the Chatbot Arena. Higher Elo scores indicate better performance in open-ended conversations.
            """)
            elo_df = get_chatbot_arena_elo_scores_df()
            st.dataframe(elo_df)
            plot_elo_scores_bar_chart(elo_df, "Chatbot Arena Elo Scores for Gemma 3 and Competitors")

        if "Zero-shot General Abilities" in selected_benchmarks:
            st.markdown("---")
            st.markdown("#### Zero-shot General Abilities")
            st.markdown("""
            Table 6 | Performance of instruction fine-tuned (IT) models compared to Gemini 1.5, Gemini 2.0, and Gemma 2 on zero-shot benchmarks across different abilities. This provides an overview of how models perform on tasks they haven't been explicitly trained on, showcasing their general intelligence.
            """)
            zero_shot_df = get_zero_shot_abilities_df()
            st.dataframe(zero_shot_df.set_index("Model"))

            st.markdown(r"""
            Figure 2 | Summary of the performance of different pre-trained models from Gemma 2 and 3 across general abilities. Radar charts effectively visualize multi-dimensional performance, making it easy to compare strengths and weaknesses across categories like Vision, Code, Science, Factuality, Reasoning, and Multilingual capabilities.
            """)
            
            # Select models for radar chart
            radar_chart_models = st.multiselect(
                "Select models for Radar Chart (Zero-shot Abilities):",
                options=zero_shot_df["Model"].unique().tolist(),
                default=zero_shot_df["Model"].unique().tolist(),
                key="radar_chart_model_selection"
            )
            
            if radar_chart_models:
                categories = [col for col in zero_shot_df.columns if col != "Model"]
                filtered_radar_df = zero_shot_df[zero_shot_df["Model"].isin(radar_chart_models)]
                if not filtered_radar_df.empty:
                    plot_radar_chart(filtered_radar_df, radar_chart_models, categories, "Zero-shot General Abilities Performance")
                else:
                    st.info("No data to display for selected models in radar chart.")
            else:
                st.info("Please select models to generate the radar chart for Zero-shot General Abilities.")


        if "Other Benchmarks (e.g., Financial QA)" in selected_benchmarks:
            st.markdown("---")
            st.markdown("#### Other Benchmarks")
            st.markdown("""
            This section presents selected additional benchmark results from the technical report (e.g., Tables 9-18), illustrating model performance on specialized tasks relevant to financial data engineering.
            """)
            other_benchmarks_df = get_other_benchmarks_df()
            st.dataframe(other_benchmarks_df.set_index("Benchmark"))

        if "Long Context Performance" in selected_benchmarks:
            st.markdown("---")
            st.markdown("#### Long Context Performance")
            st.markdown(r"""
            Figure 7 | Long context performance of pre-trained models before and after RoPE rescaling. This line chart demonstrates how perplexity (a measure of how well a probability model predicts a sample) changes as the context length increases, highlighting the models' ability to handle and understand longer documents. Lower perplexity indicates better performance.
            """)
            long_context_df = get_long_context_performance_df()
            st.dataframe(long_context_df)
            plot_long_context_line_chart(long_context_df, "Long Context Performance: Perplexity vs. Context Length")
    else:
        st.info("Select benchmarks and click 'Generate Visualizations' to see the results.")

    st.session_state["selected_performance_benchmarks"] = selected_benchmarks # Persist selections
