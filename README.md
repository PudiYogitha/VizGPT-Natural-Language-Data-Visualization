# VizGPT

VizGPT is a Python app that converts natural language descriptions into intelligent data visualizations.

## Features

- Load a sample dataset or upload your own CSV
- Enter natural language instructions like "show total sales by month" or "compare tip amounts by day and time"
- Generate a plot automatically using an LLM-driven visualization spec
- Render charts with Plotly in a Streamlit web UI

## Setup

1. Create a Python environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Ollama and the Llama 3 model:
   - Follow the install instructions at https://ollama.com/docs
   - Then install the local model:
     ```bash
     ollama pull llama3
     ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage

- Select a sample dataset or upload a CSV file.
- Type a natural language request for the visualization.
- Click "Generate Visualization" and view the chart.

## Notes

The app uses Ollama with the local `llama3` model to produce a structured chart spec from the natural language prompt and then renders the chart locally with Plotly.

