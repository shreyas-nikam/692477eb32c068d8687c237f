# 🚀 GemmaVision-QuantAdvisor: A Streamlit Application for Financial Data Engineers

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

**GemmaVision-QuantAdvisor** is a specialized Streamlit application meticulously crafted for **Financial Data Engineers**. This platform serves as an interactive hub to explore, evaluate, and compare the latest Gemma 3 models, with a keen focus on their capabilities for multimodal financial document understanding, diverse quantization strategies, and comprehensive performance benchmarks.

In the rapidly evolving landscape of financial data processing, Large Language Models (LLMs) are becoming indispensable tools for tasks ranging from automated report analysis to intelligent invoice parsing. Gemma 3, with its advanced multimodal features and improved efficiency, presents a compelling solution. This application is designed to provide Financial Data Engineers with the crucial insights needed to make informed deployment decisions, precisely tailored to their specific hardware constraints, computational requirements, and operational costs.

## Features

The GemmaVision-QuantAdvisor application offers a suite of functionalities designed to provide deep insights into the Gemma 3 model family:

*   **Gemma 3 Model Overview & Parameter Counts**:
    *   Detailed breakdown of Gemma 3 model architectures, including Vision Encoder, Embedding, and Non-embedding parameters.
    *   Tabular display of parameter counts in millions for various Gemma 3 models (1B, 4B-IT, 12B-IT, 27B-IT).
    *   Interactive stacked bar chart to visualize the distribution of parameter types across selected models.
*   **Quantization Strategies & Memory Footprints**:
    *   Explanations of key quantization concepts: bfloat16, Int4, SFP8, and KV Caching.
    *   Tabular comparison of memory footprints (in GB) for raw (bfloat16) and quantized checkpoints (Int4, SFP8), with and without KV caching, across different Gemma 3 models.
    *   Interactive bar chart to visualize memory footprints based on selected models, quantization strategies, and KV cache inclusion.
*   **Simulated Multimodal Document Understanding**:
    *   A simulated environment for Financial Data Engineers to "upload" financial documents (JPG, PNG, PDF).
    *   Interactive selection of multimodal tasks: Extract Key Figures, OCR Text Extraction, and Table Data Extraction.
    *   Simulated output for selected tasks using Gemma 3 models (e.g., extracting total amount, date, generating OCR text, structured table data).
    *   *Note: Actual Gemma 3 model inference is simulated for this lab project.*
*   **Performance Benchmarks**:
    *   **Chatbot Arena Elo Scores**: Display of Gemma 3 27B IT's standing in blind side-by-side human evaluations against other leading LLMs.
    *   **Pre-trained Model Abilities (Radar Chart)**: Interactive radar chart comparing Gemma 2 and Gemma 3 models across general abilities like Vision, Code, Science, Factuality, Reasoning, and Multilingual capabilities.
    *   **Detailed Instruction Fine-tuned (IT) Model Benchmarks**: Tabular display of zero-shot benchmark performance for Gemma 3 IT models against Gemini and Gemma 2 on various academic and domain-specific tests (e.g., MMLU-Pro, LiveCodeBench, MMMU).

## Getting Started

Follow these instructions to set up and run the GemmaVision-QuantAdvisor application on your local machine.

### Prerequisites

*   Python 3.8+ (recommended 3.9 or higher)
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/GemmaVision-QuantAdvisor.git
    cd GemmaVision-QuantAdvisor
    ```
    *(Replace `your-username` with the actual repository owner if forked)*

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    *   **On Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required Python packages**:
    ```bash
    pip install -r requirements.txt
    ```
    (If `requirements.txt` is not provided, you can generate one using `pip freeze > requirements.txt` after manually installing dependencies, or manually install them as below.)

    If `requirements.txt` is not available, install them manually:
    ```bash
    pip install streamlit pandas matplotlib seaborn numpy Pillow plotly
    ```

## Usage

Once the prerequisites are installed and the virtual environment is active, you can run the Streamlit application.

1.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```

2.  **Access the application**:
    *   After running the command, your web browser should automatically open to the Streamlit application, usually at `http://localhost:8501`.
    *   If it doesn't open automatically, copy and paste the URL provided in your terminal into your browser.

3.  **Navigate the Application**:
    *   Use the **sidebar navigation** on the left to switch between the main sections:
        *   "Gemma 3 Model Overview"
        *   "Quantization Strategies & Multimodal Understanding"
        *   "Performance Benchmarks"
    *   Interact with the various widgets (multiselect boxes, radio buttons, file uploader, checkboxes) to customize data display and analysis.

## Project Structure

The project is organized into the following directory and file structure:

```
GemmaVision-QuantAdvisor/
├── app.py                      # Main Streamlit application entry point and navigation
├── application_pages/          # Directory containing individual Streamlit pages
│   ├── __init__.py             # Makes application_pages a Python package
│   ├── page1.py                # Gemma 3 Model Overview: Parameter counts and visualization
│   ├── page2.py                # Quantization Strategies & Simulated Multimodal Understanding
│   └── page3.py                # Performance Benchmarks: Elo scores, radar charts, detailed benchmarks
├── requirements.txt            # List of Python dependencies (generate if not present)
└── README.md                   # This README file
```

## Technology Stack

*   **Application Framework**: [Streamlit](https://streamlit.io/)
*   **Programming Language**: Python
*   **Data Manipulation**: [pandas](https://pandas.pydata.org/)
*   **Numerical Operations**: [NumPy](https://numpy.org/)
*   **Data Visualization**:
    *   [Matplotlib](https://matplotlib.org/)
    *   [Seaborn](https://seaborn.pydata.org/)
    *   [Plotly Graph Objects](https://plotly.com/python/graph-objects/)
*   **Image Processing**: [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/) (for simulated document handling)

## Contributing

As this is designed as a lab project for educational and exploratory purposes, direct contributions via pull requests are generally not expected. However, feel free to fork the repository, experiment with the code, and implement your own enhancements or analyses.

If you encounter any bugs, have suggestions for improvement, or wish to discuss the project, please open an issue on the GitHub repository.

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) [Year] [Your Name or Organization Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
*(Replace `[Year]` and `[Your Name or Organization Name]` with appropriate information if you plan to publish this with your own licensing.)*

## Contact

For questions, feedback, or further discussion, please reach out to:

*   **Organization**: QuantUniversity
*   **Website**: [www.quantuniversity.com](https://www.quantuniversity.com/)
*   **GitHub Issues**: [GemmaVision-QuantAdvisor Issues](https://github.com/your-username/GemmaVision-QuantAdvisor/issues) *(Replace with actual link if available)*