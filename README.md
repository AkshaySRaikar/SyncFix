# SyncFix 🔧

An offline, CPU-deployable Multimodal RAG system for question answering over technical PDF manuals. No cloud. No GPU. No internet required.

---

## What it does

You upload a technical manual (PDF). You ask a question in plain English. SyncFix finds the most relevant sections, surfaces the associated diagrams, and generates a precise answer — all running locally on your machine.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF Extraction | PyMuPDF (fitz) |
| Chunking | Sliding window (300 words, 50 overlap) |
| Embedding | all-MiniLM-L6-v2 (Sentence-Transformers) |
| Vector Store | ChromaDB (HNSW, cosine similarity) |
| LLM Inference | Llama 3.2 1B via Ollama |
| UI | Streamlit |

---

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running


### Install
# Clone the repo
git clone https://github.com/AkshaySRaikar/syncfix.git
go to the right folder

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Pull the LLM model
ollama pull llama3.2:1b


### Run
streamlit run app.py


## Usage

1. Launch the app with `streamlit run app.py`
2. Upload your PDF from the sidebar
3. Click **Index PDF** and wait for confirmation
4. Type your question and get an answer
