import streamlit as st
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF

from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    ChatHuggingFace,
    HuggingFaceEndpoint,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
import redis

# ⬇️ .env laden
load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# 📘 PDF-Text extrahieren
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


# 🧱 Text in Chunks
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


# 🧠 Hugging Face LLM laden
def load_llm():
    return ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="deepseek-ai/DeepSeek-R1-0528",
            task="conversational",
            huggingfacehub_api_token=api_token,
            max_new_tokens=512,
            do_sample=False,
        ),
        verbose=False,
    )


# 📦 Redis + Memory
def get_memory():
    history = RedisChatMessageHistory(session_id="streamlit_session")
    return ConversationBufferMemory(chat_memory=history, return_messages=True)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# 🔍 Retriever bauen (FAISS)
def build_retriever(texts):
    db = FAISS.from_texts(texts, embedding=embeddings)
    return db.as_retriever()


# --- Multi-Agenten ---


# 1. Intent-Analyzer: erkennt Datenquelle anhand Frage
def intent_analyzer(state):
    question = state["question"].lower()
    if "finanz" in question or "aktie" in question:
        state["data_source"] = "financial_reports"
    elif "nachricht" in question or "news" in question:
        state["data_source"] = "news_api"
    else:
        state["data_source"] = "tech_docs"
    return state


# 2. Retriever Agent: holt Dokumente aus gewählter Quelle
def retriever_agent(state):
    data_source = state.get("data_source", "tech_docs")
    query = state["question"]

    if data_source == "financial_reports":
        docs = retriever_financial.invoke(query)
    elif data_source == "news_api":
        docs = retriever_news.invoke(query)
    else:
        docs = retriever_tech.invoke(query)
    state["docs"] = docs
    return state


# 3. Generator Agent: erstellt dynamischen Prompt und generiert Antwort
def generator_agent(state):
    question = state["question"]
    docs = state.get("docs", [])
    data_source = state.get("data_source", "tech_docs")

    chat_history = memory.load_memory_variables({}).get("history", [])
    history_text = "\n".join(
        ("User: " if msg.type == "human" else "Assistant: ") + msg.content
        for msg in chat_history
    )
    context = "\n\n".join(doc.page_content for doc in docs)

    if data_source == "financial_reports":
        prompt = (
            f"Du bist ein Finanzexperte.\n"
            f"Chatverlauf:\n{history_text}\n"
            f"Kontext:\n{context}\n"
            f"Beantworte die Frage präzise:\n{question}"
        )
    elif data_source == "news_api":
        prompt = (
            f"Du bist ein Nachrichtenassistent.\n"
            f"Chatverlauf:\n{history_text}\n"
            f"Kontext:\n{context}\n"
            f"Fasse kurz zusammen:\n{question}"
        )
    else:
        prompt = (
            f"Du bist ein technischer Assistent.\n"
            f"Chatverlauf:\n{history_text}\n"
            f"Kontext:\n{context}\n"
            f"Antwort auf die technische Frage:\n{question}"
        )

    response = llm.invoke(prompt)
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(response.content)

    state["answer"] = response.content
    return state


redis_client = redis.Redis(host="localhost", port=6379, db=0)


def feedback_agent(state):
    feedback = state.get("feedback", None)
    if feedback is not None:
        # Feedback in Redis speichern, z.B. als Liste appenden
        redis_client.rpush("feedback_list", str(feedback))

        # Beispiel: Prompt anpassen bei schlechtem Feedback
        if int(feedback) < 3:
            state["prompt_tone"] = "freundlich und ausführlich"
        else:
            state["prompt_tone"] = "präzise und knapp"

    state["last_feedback"] = feedback
    return state


# Multi-Agenten Graph bauen
def build_multi_agent_app():
    graph = StateGraph(state_schema=dict)

    graph.add_node("intent_analyzer", RunnableLambda(intent_analyzer))
    graph.add_node("retriever", RunnableLambda(retriever_agent))
    graph.add_node("generator", RunnableLambda(generator_agent))
    graph.add_node("feedback", RunnableLambda(feedback_agent))

    graph.set_entry_point("intent_analyzer")

    graph.add_edge("intent_analyzer", "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "feedback")
    graph.add_edge("feedback", END)

    return graph.compile()


# --- Streamlit UI ---

st.title("🔍 Multi-Agenten RAG mit LangGraph auf PDFs")

uploaded_file = st.file_uploader("📄 PDF hochladen", type=["pdf"])

if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)
    st.success("📄 PDF geladen!")

    if "agent_app" not in st.session_state:
        # Text chunken
        chunks = chunk_text(raw_text)

        # Mehrere Retriever vorbereiten (Dummy: alle gleich, hier kannst du eigene Datenquellen definieren)
        global retriever_tech, retriever_financial, retriever_news
        retriever_tech = build_retriever(chunks)
        retriever_financial = build_retriever(chunks)
        retriever_news = build_retriever(chunks)

        global memory, llm
        memory = get_memory()
        llm = load_llm()

        st.session_state.agent_app = build_multi_agent_app()
        st.session_state.ready = True
else:
    # Wenn keine Datei hochgeladen ist, Status zurücksetzen
    st.session_state.ready = False
    if "agent_app" in st.session_state:
        del st.session_state["agent_app"]

# Frageingabe und Antwort nur anzeigen, wenn PDF verarbeitet wurde
if st.session_state.get("ready", False):
    question = st.text_input("❓ Frage stellen:")

    if question:
        with st.spinner("⏳ Antwort wird generiert..."):
            result = st.session_state.agent_app.invoke({"question": question})
            st.success("✅ Antwort:")
            st.markdown(result["answer"])
else:
    st.info("Bitte lade zuerst ein PDF hoch, um Fragen zu stellen.")
