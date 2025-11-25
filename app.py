
import streamlit as st
st.set_page_config(page_title="GemmaVision-QuantAdvisor", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("GemmaVision-QuantAdvisor")
st.divider()

st.markdown("""
In this lab, we will explore the GemmaVision-QuantAdvisor application, designed for Financial Data Engineers to evaluate and compare Gemma 3 models. This platform offers insights into multimodal capabilities, quantization strategies, and performance benchmarks, facilitating informed deployment decisions for financial document understanding tasks.
""")

page = st.sidebar.selectbox(label="Navigation", options=["Home", "Model Overview", "Quantization Strategies", "Multimodal Document Understanding", "Performance Benchmarks"])

if page == "Home":
    from application_pages.home_page import main
    main()
elif page == "Model Overview":
    from application_pages.model_overview import main
    main()
elif page == "Quantization Strategies":
    from application_pages.quantization_strategies import main
    main()
elif page == "Multimodal Document Understanding":
    from application_pages.multimodal_document_understanding import main
    main()
elif page == "Performance Benchmarks":
    from application_pages.performance_benchmarks import main
    main()
