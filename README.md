# GemmaVision-QuantAdvisor: A Streamlit Application Lab for Gemma 3 Model Evaluation

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Description

The **GemmaVision-QuantAdvisor** Streamlit application serves as an interactive platform designed for **Financial Data Engineers** to explore, evaluate, and compare Gemma 3 models. This lab project highlights Gemma 3's multimodal capabilities for financial document understanding, different quantization strategies, and performance benchmarks. The goal is to aid in informed deployment decisions tailored to specific hardware constraints and operational costs within financial data engineering workflows.

Upon using this application, users will be able to:
*   Understand the architectural and performance characteristics of Gemma 3 models.
*   Evaluate the impact of different quantization techniques on memory footprint and efficiency.
*   Compare Gemma 3's performance in multimodal tasks, general intelligence, math, and reasoning against previous Gemma versions and other state-of-the-art models.
*   Utilize quantitative benchmarks and visualizations to guide strategic model deployment in financial data engineering workflows.

## Features

The application is structured into several interactive pages, each focusing on a specific aspect of Gemma 3 model evaluation:

1.  ### Gemma 3 Model Overview
    *   **Parameter Counts Analysis**: Displays the breakdown of vision encoder, embedding, and non-embedding parameters for different Gemma 3 models.
    *   **Visual Representation**: Provides a stacked bar chart to visualize the parameter distribution, aiding in understanding model complexity and resource requirements.

2.  ### Quantization Strategies & Memory Footprint
    *   **Quantization Concepts**: Explains key quantization techniques (bfloat16, Int4, SFP8) and their impact on model memory.
    *   **Memory Footprint Comparison**: Presents a data table and an interactive bar chart comparing memory usage across various Gemma 3 models and quantization strategies, including the impact of KV caching.

3.  ### Multimodal Document Understanding
    *   **Simulated Financial Document Processing**: Allows users to "upload" (conceptually, for simulation) financial documents (JPG, PNG, PDF).
    *   **Interactive Task Selection**: Users can select simulated multimodal tasks like "Extract Key Figures," "OCR Text Extraction," or "Table Data Extraction" using different Gemma 3 models.
    *   **Simulated Outputs**: Demonstrates example outputs for each task, illustrating the potential of Gemma 3 in financial document analysis. **(Note: Actual model inference is simulated for this lab project.)**

4.  ### Performance Benchmarks
    *   **Chatbot Arena Elo Scores**: Displays a table comparing Gemma 3 with other LLMs based on human evaluation Elo scores.
    *   **Pre-trained Model Abilities Radar Chart**: An interactive radar chart comparing Gemma 2 and Gemma 3 models across general abilities (Vision, Code, Science, Factuality, Reasoning, Multilingual).
    *   **Detailed Instruction Fine-tuned (IT) Model Benchmarks**: Provides a comprehensive table showcasing the performance of instruction-tuned models across various zero-shot benchmarks.

## Getting Started

Follow these instructions to set up and run the GemmaVision-QuantAdvisor application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/gemma-vision-quantadvisor.git
    cd gemma-vision-quantadvisor
    ```
    *(Replace `your-username` with the actual repository owner's username or URL if publicly hosted.)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**
    Create a `requirements.txt` file in the root directory with the following content:

    ```
    streamlit>=1.30.0
    pandas>=2.0.0
    matplotlib>=3.7.0
    seaborn>=0.13.0
    numpy>=1.24.0
    Pillow>=10.0.0
    plotly>=5.18.0
    ```

    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**
    Ensure your virtual environment is activated, then run:
    ```bash
    streamlit run app.py
    ```

2.  **Access the application:**
    Your web browser should automatically open to `http://localhost:8501` (or a similar address) where you can interact with the GemmaVision-QuantAdvisor application.

3.  **Navigate through the sections:**
    Use the sidebar navigation to switch between the "Gemma 3 Model Overview", "Quantization Strategies & Memory Footprint", "Multimodal Document Understanding", and "Performance Benchmarks" pages.

## Project Structure

The project is organized into a modular structure for clarity and maintainability:

```
gemma-vision-quantadvisor/
├── app.py                      # Main Streamlit application entry point
├── application_pages/          # Directory containing individual page modules
│   ├── __init__.py             # Initializes the package
│   ├── page1.py                # Gemma 3 Model Overview logic
│   ├── page2.py                # Quantization Strategies & Memory Footprint logic
│   ├── page3.py                # Multimodal Document Understanding logic (simulated)
│   └── page4.py                # Performance Benchmarks logic
├── requirements.txt            # List of Python dependencies
└── README.md                   # This README file
```

## Technology Stack

*   **Streamlit**: The core framework for building the interactive web application.
*   **Pandas**: Used for data manipulation and tabular data representation.
*   **Matplotlib**: Utilized for static data visualizations, such as bar charts for parameter counts.
*   **Seaborn**: Built on Matplotlib, used for enhanced statistical data visualizations, particularly for memory footprint comparisons.
*   **NumPy**: Fundamental package for numerical operations, used indirectly by Pandas and other libraries.
*   **Pillow (PIL)**: For simulated image handling in the multimodal section.
*   **Plotly**: For advanced interactive visualizations, specifically the radar chart in the performance benchmarks.
*   **Python 3**: The primary programming language.

## Contributing

Contributions to the GemmaVision-QuantAdvisor project are welcome! If you have suggestions for improvements, bug fixes, or new features, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and ensure the code adheres to existing style.
4.  Write clear, concise commit messages.
5.  Push your branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file (if present) or refer to the repository for details.

## Contact

For questions, feedback, or further information about this lab project, please reach out to:

*   **QuantUniversity**: [info@quantuniversity.com](mailto:info@quantuniversity.com)
*   **Project Repository**: [https://github.com/your-username/gemma-vision-quantadvisor](https://github.com/your-username/gemma-vision-quantadvisor) *(Replace with actual URL)*

---

## References

The data and concepts presented in this application are derived from the following sources, primarily the technical reports on Gemma models:

*   **[1]** Gemma Family of Models (e.g., Gemma 2, Gemma 3) technical reports and documentation.
*   **[2]** Specific tables and figures referenced directly within the application code (e.g., Table 1 for parameter counts, Figure 2 for model abilities).
*   **[3]** Specific tables and figures referenced directly within the application code (e.g., Table 3 for quantization memory footprints, Table 5 for Chatbot Arena scores, Table 6 for IT model benchmarks).

*(Note: Specific URLs for the Gemma technical reports can be added here if available and appropriate.)*
