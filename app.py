import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image # For simulated image handling
import math # For mathematical constants, e.g., for radar charts
import plotly.graph_objects as go # For advanced charts like radar charts

st.set_page_config(page_title="GemmaVision-QuantAdvisor", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("GemmaVision-QuantAdvisor")
st.divider()

st.markdown("""
In this lab, the GemmaVision-QuantAdvisor Streamlit application serves as an interactive platform for **Financial Data Engineers** to explore, evaluate, and compare Gemma 3 models. It highlights their multimodal capabilities for financial document understanding, different quantization strategies, and performance benchmarks to aid in informed deployment decisions tailored to specific hardware constraints and operational costs.

Upon using this application, users will be able to:
-   Understand the architectural and performance characteristics of Gemma 3 models.
-   Evaluate the impact of different quantization techniques on memory footprint and efficiency.
-   Compare Gemma 3's performance in multimodal tasks, general intelligence, math, and reasoning against previous Gemma versions and other state-of-the-art models.
-   Utilize quantitative benchmarks and visualizations to guide strategic model deployment in financial data engineering workflows.
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
