import streamlit as st
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
from typing import TypedDict, List
import redis

# LangChain & LangGraph Imports
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
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

# Agent-specific imports
from transformers import pipeline
from duckduckgo_search import DDGS

# --- Configuration & Setup ---

# Load environment variables from .env file
load_dotenv()
API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Check for API Token
if not API_TOKEN:
    st.error("HUGGINGFACEHUB_API_TOKEN not found in .env file. Please add it.")
    st.stop()

# --- Helper Functions ---


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text from a PDF file."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def chunk_text(text: str) -> List[str]:
    """Splits text into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)


@st.cache_resource
def get_embeddings_model():
    """Loads the sentence-transformer model for embeddings."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_llm():
    """Loads the language model from Hugging Face."""
    return ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation",
            huggingfacehub_api_token=API_TOKEN,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        ),
        verbose=False,
    )


@st.cache_resource
def get_classifier():
    """Loads the zero-shot classification model."""
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def get_memory() -> ConversationBufferMemory:
    """Initializes conversation memory with Redis backend."""
    try:
        # Ensure Redis is reachable
        redis_client = redis.Redis(host="localhost", port=6379, db=0)
        redis_client.ping()
        history = RedisChatMessageHistory(session_id="streamlit_chat_session")
        return ConversationBufferMemory(
            chat_memory=history, return_messages=True, memory_key="chat_history"
        )
    except redis.exceptions.ConnectionError:
        st.error("Could not connect to Redis. Chat history will not be saved.")
        # Fallback to in-memory if Redis is not available
        return ConversationBufferMemory(return_messages=True, memory_key="chat_history")


# --- LangGraph Agent State Definition ---


class AgentState(TypedDict):
    """Defines the state of our agent graph."""

    question: str
    answer: str
    intent: str
    data_source: str
    docs: List[Document]
    chat_history: List[BaseMessage]


# --- Agent Nodes for the Graph ---


def intent_analyzer_node(state: AgentState) -> AgentState:
    """
    Analyzes the user's question to determine intent and potential data source.
    This is the first step to decide the workflow.
    """
    st.write("🕵️‍♂️ **Agent: Intent Analyzer** - Analyzing question...")
    question = state["question"]
    classifier = get_classifier()

    # Determine the primary intent
    intent_labels = ["question_answering", "summary_request", "internet_search"]
    intent_result = classifier(question, intent_labels)
    intent = intent_result["labels"][0]
    state["intent"] = intent
    st.write(f"🔍 Intent Detected: **{intent}**")

    # Determine the likely data source (for PDF-based tasks)
    if intent in ["question_answering", "summary_request"]:
        source_labels = ["financial_reports", "news_api", "tech_docs"]
        source_result = classifier(question, source_labels)
        state["data_source"] = source_result["labels"][0]
        st.write(f"📚 Data Source Hint: **{state['data_source']}**")

    return state


def retriever_node(state: AgentState) -> AgentState:
    """
    Retrieves relevant documents from the appropriate vector store based on the question.
    """
    st.write("📚 **Agent: Retriever** - Searching PDF documents...")
    question = state["question"]
    data_source = state["data_source"]

    # In a real-world scenario, you would have different retrievers for different sources.
    # Here, we simulate this by selecting a retriever, but they all point to the same DB.
    if data_source == "financial_reports":
        retriever = st.session_state.retriever_financial
    elif data_source == "news_api":
        retriever = st.session_state.retriever_news
    else:  # Default to tech_docs
        retriever = st.session_state.retriever_tech

    docs = retriever.invoke(question)
    state["docs"] = docs
    return state


def web_search_node(state: AgentState) -> AgentState:
    """
    Performs a web search using DuckDuckGo for questions requiring up-to-date information.
    """
    st.write("🌐 **Agent: Web Searcher** - Searching the internet...")
    question = state["question"]

    with DDGS() as ddgs:
        results = list(ddgs.text(question, max_results=3))
        web_texts = [res.get("body", "") for res in results]

    # Format results as LangChain Documents
    docs = [Document(page_content=text) for text in web_texts if text]
    state["docs"] = docs
    return state


def generator_node(state: AgentState) -> AgentState:
    """
    Generates the final answer based on the retrieved context, chat history, and question.
    """
    st.write("✍️ **Agent: Generator** - Formulating the answer...")
    question = state["question"]
    docs = state.get("docs", [])
    intent = state["intent"]
    memory = get_memory()
    llm = load_llm()

    # Load history and format it
    chat_history = memory.load_memory_variables({}).get("chat_history", [])

    context = "\n\n".join(doc.page_content for doc in docs)

    # Dynamically create the prompt based on intent
    if intent == "summary_request":
        prompt_template = (
            "You are a helpful assistant. Summarize the following context based on the user's request.\n"
            "Context:\n{context}\n\nUser's Request: {question}"
        )
    elif intent == "internet_search":
        prompt_template = (
            "You are an informative assistant. Answer the user's question using the provided web search results.\n"
            "Web Search Results:\n{context}\n\nQuestion: {question}"
        )
    else:  # Default to question_answering
        prompt_template = (
            "You are a helpful Q&A assistant. Use the following context from the document to answer the user's question. "
            "If the answer is not in the context, say so.\n"
            "Context:\n{context}\n\nQuestion: {question}"
        )

    # Create the generation chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
        {"context": lambda x: context, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Invoke the chain and update memory
    answer = rag_chain.invoke(question)
    memory.save_context({"input": question}, {"output": answer})

    state["answer"] = answer
    return state


# --- Graph Definition & Conditional Logic ---


def router(state: AgentState) -> str:
    """
    Conditional logic to route the workflow based on the detected intent.
    This is the core of making the intent analyzer useful.
    """
    st.write("🚦 **Router** - Deciding next step...")
    intent = state["intent"]
    if intent == "internet_search":
        st.write("Route -> Web Search")
        return "web_search"
    else:
        st.write("Route -> Document Retrieval")
        return "retrieve"


def build_multi_agent_graph() -> StateGraph:
    """Builds and compiles the LangGraph agent graph."""
    graph = StateGraph(AgentState)

    # Add nodes to the graph
    graph.add_node("intent_analyzer", intent_analyzer_node)
    graph.add_node("retrieve", retriever_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generator_node)

    # Define the workflow edges
    graph.set_entry_point("intent_analyzer")

    # This is the conditional edge. It calls the `router` function to decide the next node.
    graph.add_conditional_edges(
        "intent_analyzer",
        router,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
        },
    )

    # After retrieving from the PDF or searching the web, the flow converges to the generator.
    graph.add_edge("retrieve", "generate")
    graph.add_edge("web_search", "generate")

    # The generator is the final step.
    graph.add_edge("generate", END)

    return graph.compile()


# --- Streamlit UI ---

st.set_page_config(page_title="Multi-Agent RAG with LangGraph", layout="wide")
st.title("🔍 Multi-Agent RAG with LangGraph on PDFs")

# Initialize session state
if "agent_app" not in st.session_state:
    st.session_state.agent_app = None
    st.session_state.ready = False
    st.session_state.messages = []

# Sidebar for PDF upload and setup
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

    if uploaded_file and st.button("Process PDF and Build Agents"):
        with st.spinner(
            "Processing PDF, building vector store, and compiling agents..."
        ):
            file_bytes = uploaded_file.getvalue()
            raw_text = extract_text_from_pdf(file_bytes)

            if raw_text:
                chunks = chunk_text(raw_text)
                embeddings = get_embeddings_model()

                # Build the FAISS vector store
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)

                # --- IMPORTANT ---
                # For this demo, all retrievers use the same vector store.
                # In a real app, you'd load different data into each.
                st.session_state.retriever_tech = vector_store.as_retriever(
                    search_kwargs={"k": 3}
                )
                st.session_state.retriever_financial = vector_store.as_retriever(
                    search_kwargs={"k": 3}
                )
                st.session_state.retriever_news = vector_store.as_retriever(
                    search_kwargs={"k": 3}
                )

                # Build and compile the graph
                st.session_state.agent_app = build_multi_agent_graph()

                st.session_state.ready = True
                st.session_state.messages = []  # Clear chat history on new file
                st.success("✅ PDF processed and agents are ready!")
            else:
                st.error("Could not extract text from the PDF.")

# Main chat interface
if not st.session_state.ready:
    st.info("Please upload a PDF and click 'Process PDF' in the sidebar to begin.")
else:
    st.success("Agents are ready! Ask a question below.")

    # Display chat history
    memory = get_memory()
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    for msg in chat_history:
        st.chat_message("human" if msg.type == "human" else "ai").write(msg.content)

    # User input
    if question := st.chat_input(
        "Ask a question about the document, or ask for a web search..."
    ):
        st.chat_message("human").write(question)

        with st.spinner("🤖 Agents are thinking..."):
            # The right side will show the agent's internal monologue/steps
            with st.expander("Agent Workflow Steps"):
                # Invoke the agent graph
                result_state = st.session_state.agent_app.invoke({"question": question})

            st.chat_message("ai").write(result_state["answer"])
