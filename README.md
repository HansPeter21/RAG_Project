# 🤖 Multi-Agent RAG System mit LangGraph, FAISS & Streamlit

Dieses Projekt implementiert ein **Multi-Agent Retrieval-Augmented Generation (RAG) System**, das mit **LangGraph** orchestriert wird. Es verwendet zwei spezialisierte Agenten zur Verarbeitung natürlicher Sprache, unterstützt **Memory** mit Redis, verarbeitet **PDF-Dokumente** und bietet eine benutzerfreundliche Oberfläche mit **Streamlit**.

---

## ✅ Projektziele

### 1. Implementierung eines Multi-Agenten-RAG-Systems

- [x] **RetrieverAgent** zur semantischen Suche relevanter Inhalte aus Dokumenten
- [x] **GeneratorAgent** zur Erzeugung von Antworten auf Benutzerfragen
- [x] **Frei wählbare Dokumentbasis** (z. B. technische Publikationen, Nachrichten, Finanzberichte)
- [x] **Zusammenfassungsfunktion** langer Dokumente (via Prompt Engineering möglich)
- [x] **Benutzeroberfläche mit Streamlit**
  - [x] PDF-Upload
  - [x] Interaktive Frage-Antwort-Funktion
  - [x] Ausgabe der generierten Antworten
- [x] Kein Guardrails-System notwendig (z. B. Moderation, Filterung)

---

### 2. Memory-Integration

- [x] **Persistente Speicherung** von Chatverläufen mit Redis
- [x] Integration mit `LangChain` Memory-System

---

### 3. Autonomes Entscheidungsmanagement (fortgeschrittenes Ziel)

- [ ] Agenten analysieren die semantische Struktur der Nutzeranfrage
- [ ] Dynamische Auswahl von Datenquellen
- [ ] Anpassung der Prompts auf Basis von Feedback (z. B. Follow-up-Fragen, Bewertung)

---

## ⚙️ Technische Architektur

| Komponente               | Beschreibung                                                                 |
|--------------------------|------------------------------------------------------------------------------|
| `LangGraph`              | Graphbasierte Orchestrierung für Agenten-Workflows mit Schleifen & Feedback |
| `RetrieverAgent`         | Führt semantische Suche mit FAISS aus                                        |
| `GeneratorAgent`         | Verwendet HuggingFace LLM zur Antwortgenerierung                             |
| `FAISS`                  | Vektorsuche für eingebettete Dokumente                                       |
| `Redis`                  | Speicherung von Chatverlauf für Kontextwahrung                               |
| `Streamlit`              | Weboberfläche für Upload, Frage & Antwort                                    |
| `dotenv`                 | Umgebungsvariablen (API-Schlüssel)                                           |

---

## 📦 Setup

1. **.env Datei erstellen**

```env
HUGGINGFACEHUB_API_TOKEN=hf_...
