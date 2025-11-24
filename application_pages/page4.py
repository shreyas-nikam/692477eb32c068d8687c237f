import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def run_page4():
    st.markdown("### Performance Benchmarks")
    st.markdown("""
    This section provides a comparative overview of Gemma 3 models against other state-of-the-art models on various benchmarks, including Chatbot Arena scores, zero-shot benchmarks for general abilities, and multimodal performance.
    """)

    st.subheader("Chatbot Arena Elo Scores (Table 5 equivalent)")
    st.markdown("""
    Evaluation of Gemma 3 27B IT model in the Chatbot Arena based on blind side-by-side evaluations by human raters. Scores are based on the Elo rating system. (Data from Table 5 of the technical report).
    """)

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

    st.markdown("""
    Overall, the radar charts indicate that newer Gemma 3 versions generally show improved performance across most categories, with a notable enhancement in vision capabilities due to their multimodal architecture.
    """)

    st.markdown("---")
    st.subheader("Detailed Instruction Fine-tuned (IT) Model Benchmarks")
    st.markdown("""
    Performance of instruction fine-tuned (IT) models compared to Gemini 1.5, Gemini 2.0, and Gemma 2 on zero-shot benchmarks across different abilities (Data from Table 6 of the technical report).
    """)

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