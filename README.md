# Media Bias Analysis System

## Overview
This project is a full-stack application designed to detect and analyze bias in news articles. It utilizes advanced Natural Language Processing (NLP) techniques and Explainable AI (XAI) to provide users with deep insights into *why* a specific text might be considered biased.

## Key Features
*   **Bias Detection**: Uses a fine-tuned RoBERTa-based model (`himel7/bias-detector`) to classify text as "Biased" or "Non-biased".
*   **Explainable AI (XAI)**:
    *   **SHAP (SHapley Additive exPlanations)**: Visualizes the "push and pull" contribution of each word to the final score.
    *   **LIME (Local Interpretable Model-agnostic Explanations)**: Identifies local trigger words that influence the prediction.
    *   **Attention Mechanism**: Visualizes the raw self-attention weights from the model's last layer, showing exactly where the model "looked".
*   **AI Narrative Analysis**: Generates a human-readable explanation of the bias using **Granite 4:3b** (via Ollama), grounded in the mathematical evidence from SHAP/LIME/Attention.
*   **Interactive UI**: A modern, responsive React frontend with interactive heatmaps, charts, and on-demand analysis.

## Tech Stack

### Backend
*   **Framework**: FastAPI (Python)
*   **ML/AI**: PyTorch, Transformers (Hugging Face), SHAP, LIME
*   **LLM Integration**: Ollama (running Granite 4:3b)
*   **Data Processing**: Pandas, NumPy

### Frontend
*   **Framework**: React (Vite)
*   **Styling**: Tailwind CSS
*   **Visualizations**: Chart.js, Framer Motion
*   **Icons**: Lucide React

## Project Structure

```
Mini_Project/
├── backend/                 # FastAPI Backend
│   ├── main.py             # API Entry point & Logic
│   ├── bias_detector.py    # ML Model wrapper & XAI logic
│   └── requirements.txt    # Python dependencies
├── react_frontend/          # React Frontend
│   ├── src/
│   │   ├── App.jsx         # Main UI Component
│   │   └── index.css       # Global Styles (Tailwind)
│   ├── package.json        # Node dependencies
│   └── vite.config.js      # Vite configuration
├── start_project.sh         # Automated startup script
└── README.md               # This file
```

## Prerequisites
Before running the project, ensure you have the following installed:
1.  **Python 3.10+**
2.  **Node.js & npm**
3.  **Ollama**: Required for the narrative analysis.
    *   Install from [ollama.com](https://ollama.com)
    *   Ensure the service is running (`systemctl start ollama` or launch the app).

## Setup & Installation

### Automated Setup
We provide a script to handle the entire setup process (venv creation, dependency installation, model pulling, and server startup).

1.  Make the script executable:
    ```bash
    chmod +x start_project.sh
    ```
2.  Run the script:
    ```bash
    ./start_project.sh
    ```

### Manual Setup

#### 1. Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Pull the LLM model
ollama pull granite4:3b
# Start the server
uvicorn main:app --reload --port 8000
```

#### 2. Frontend Setup
```bash
cd react_frontend
npm install
npm run dev
```

## Usage
1.  Open your browser to `http://localhost:5173`.
2.  Paste a news article or text snippet into the input box.
3.  Select your explanation mode:
    *   **SHAP**: Most accurate, slower.
    *   **LIME**: Good local approximation, faster than SHAP.
    *   **Attention**: Instant visualization of model focus.
4.  Click **"Detect Bias"**.
5.  View the Verdict, Confidence Score, and Visualizations.
6.  Click **"Explain with AI"** to generate a detailed narrative report.

## Troubleshooting
*   **CUDA Errors**: If you encounter GPU memory issues, the backend automatically truncates text to 500 tokens. Restart the backend if a hard crash occurs.
*   **Ollama Connection**: Ensure Ollama is running on port 11434.
*   **Missing Dependencies**: Re-run the installation commands or the startup script.
