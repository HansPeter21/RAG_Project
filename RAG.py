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


# 🧠 Hugging Face LLM
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


# 🔍 Retriever
def build_retriever(texts):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.from_texts(texts, embedding=embeddings)
    return db.as_retriever()


# 🔄 LangGraph
def build_app(retriever, llm, memory):
    def retrieve(state):
        query = state["question"]
        docs = retriever.invoke(query)
        state["docs"] = docs
        return state

    def generate(state):
        query = state["question"]
        docs = state["docs"]
        chat_history = memory.load_memory_variables({}).get("history", [])
        history_text = "\n".join(
            ("User: " if msg.type == "human" else "Assistant: ") + msg.content
            for msg in chat_history
        )
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = (
            f"Du bist ein hilfreicher Assistent.\n"
            f"Hier ist der bisherige Chatverlauf:\n{history_text}\n"
            f"Basierend auf folgendem Kontext beantworte die Frage:\n{context}\n\n"
            f"Frage: {query}"
        )
        response = llm.invoke(prompt)
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(response.content)
        state["answer"] = response.content
        return state

    graph = StateGraph(state_schema=dict)
    graph.add_node("retriever", RunnableLambda(retrieve))
    graph.add_node("llm", RunnableLambda(generate))
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "llm")
    graph.add_edge("llm", END)
    return graph.compile()


# 🌐 Streamlit UI
st.title("🔍 RAG + LangGraph auf PDFs")

uploaded_file = st.file_uploader("📄 PDF hochladen", type=["pdf"])

if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)
    st.success("📄 PDF geladen!")

    if "retriever" not in st.session_state:
        chunks = chunk_text(raw_text)
        retriever = build_retriever(chunks)
        memory = get_memory()
        llm = load_llm()
        app = build_app(retriever, llm, memory)
        st.session_state.app = app
        st.session_state.ready = True

if st.session_state.get("ready", False):
    question = st.text_input("❓ Frage stellen:")

    if question:
        with st.spinner("⏳ Antwort wird generiert..."):
            result = st.session_state.app.invoke({"question": question})
            st.success("✅ Antwort:")
            st.markdown(result["answer"])
