# 🧠 Multi-Agent RAG mit LangGraph, Streamlit & Huggingface

Dieses Projekt ist eine interaktive Streamlit-Anwendung, die eine Multi-Agenten-RAG-Architektur verwendet, um Fragen zu PDF-Dokumenten zu beantworten oder bei Bedarf eine Websuche durchzuführen. Die Agentenlogik basiert auf [LangGraph](https://github.com/langchain-ai/langgraph).

## 🚀 Features

- 📄 PDF-Upload & -Verarbeitung mit `PyMuPDF`
- 🔎 Semantische Vektorsuche mit FAISS & Huggingface Embeddings
- 🧠 Intenterkennung via Zero-Shot-Classifikation
- 🌐 Live-Websuche mit `duckduckgo_search`
- 🤖 Generierung von Antworten über Huggingface LLMs (z. B. Mistral-7B)
- 🧵 Chatverlauf (persistiert in Redis, falls verfügbar)

## 📦 Abhängigkeiten

- Python ≥ 3.10  
- `.env`-Datei mit Huggingface API-Token

### 🛠️ Installation

```bash
git clone https://github.com/dein-benutzername/langgraph-multiagent-rag.git
cd langgraph-multiagent-rag
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt