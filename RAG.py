from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory
import redis
import os

# [Frage vom Nutzer]
#    ↓
# [Retriever holt relevante Texte]
#    ↓
# [LLM von Hugging Face erzeugt Antwort basierend auf Kontext]
#    ↓
# [Antwort an Nutzer]

# 🔐 API-Key für Hugging Face setzen (oder als Umgebungsvariable exportieren)
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 📘 1. Beispieltexte vorbereiten (diese würden aus PDF oder Ähnlichem kommen)
texts = [
    "RAG steht für Retrieval-Augmented Generation und kombiniert Suche mit LLM.",
    "LangGraph ist ein Tool zur Orchestrierung von KI-Workflows auf Graphbasis.",
    "FAISS ist eine Bibliothek zur schnellen Ähnlichkeitssuche auf Vektoren.",
]

# Text in kleinere Abschnitte aufteilen
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
for text in texts:
    chunks.extend(text_splitter.split_text(text))

# 📐 2. Embeddings + Retriever vorbereiten
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_texts(chunks, embedding=embeddings)

retriever = db.as_retriever()

# 🔴 Redis-Verbindung aufbauen
redis_client = redis.Redis(host="localhost", port=6379, db=0)
history = RedisChatMessageHistory(session_id="session_1")
memory = ConversationBufferMemory(chat_memory=history, return_messages=True)

# 🧠 4. LLM von Hugging Face konfigurieren
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1-0528",
        task="conversational",
        huggingfacehub_api_token=api_token,
        max_new_tokens=512,
        do_sample=False,
    ),
    verbose=False,
)


# 🔍 5. Retriever-Funktion
def retrieve(state):
    query = state["question"]
    docs = retriever.invoke(query)
    state["docs"] = docs
    return state


# ✍️ 6. Antwort generieren mit Memory-Integration
def generate_answer(state):
    query = state["question"]
    docs = state["docs"]

    # Chat-Verlauf aus Memory laden (Liste von Messages)
    chat_history = memory.load_memory_variables({}).get("history", [])

    # Chat-Verlauf als Text (User + Assistant Nachrichten)
    history_text = ""
    for msg in chat_history:
        role = msg.type  # z.B. "human" oder "ai"
        content = msg.content
        prefix = "User:" if role == "human" else "Assistant:"
        history_text += f"{prefix} {content}\n"

    # Kontext aus Dokumenten
    context = "\n\n".join([doc.page_content for doc in docs])

    # Prompt mit Chat-Historie, Kontext und Frage
    prompt = (
        f"Du bist ein hilfreicher Assistent.\n"
        f"Hier ist der bisherige Chatverlauf:\n{history_text}\n"
        f"Basierend auf folgendem Kontext beantworte die Frage:\n{context}\n\n"
        f"Frage: {query}"
    )

    answer = llm.invoke(prompt)

    # User-Message + Antwort in Memory speichern
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(answer.content)

    state["answer"] = answer.content
    return state


# 🧠 7. LangGraph erstellen
graph = StateGraph(state_schema=dict)
graph.add_node("retriever", RunnableLambda(retrieve))
graph.add_node("llm", RunnableLambda(generate_answer))
graph.set_entry_point("retriever")
graph.add_edge("retriever", "llm")
graph.add_edge("llm", END)

# 🧵 8. Kompilieren und verwenden
app = graph.compile()

# 🚀 9. Testaufruf
frage = "Was ist LangGraph?"
result = app.invoke({"question": frage})
print("Antwort:\n", result["answer"])
