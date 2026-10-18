

import re
import requests
import streamlit as st
import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer

# =========================
# 1. Configuration
# =========================
from pathlib import Path

PDF_PATH = Path(__file__).parent / "scholarship_fixed.pdf"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_NAME = "qwen2.5:1.5b"

st.set_page_config(
    page_title="Scholarship AI Assistant",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Scholarship AI Assistant")
st.caption(
    "Ask questions about the Heydar Aliyev scholarship document."
)


# =========================
# 2. Extract PDF sections
# =========================
def extract_section(text, start_marker, end_markers):
    start = text.find(start_marker)

    if start == -1:
        return ""

    positions = [
        text.find(marker, start + len(start_marker))
        for marker in end_markers
    ]
    positions = [p for p in positions if p != -1]

    end = min(positions) if positions else len(text)
    return text[start:end].strip()


@st.cache_resource(show_spinner="Loading the PDF and AI model...")
def initialize_rag():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    with pymupdf.open(PDF_PATH) as pdf:
        full_text = "\n".join(page.get_text() for page in pdf)

    if len(full_text.strip()) < 100:
        raise ValueError("Could not extract enough text from the PDF.")

    benefits = extract_section(
        full_text,
        "3. SCHOLARSHIP BENEFITS",
        ["6. APPLICATION PROCESS", "7. REQUIREMENTS"],
    )

    requirements = extract_section(
        full_text,
        "7. REQUIREMENTS",
        ["7.1 PREREQUISITE FOR THE DOCUMENTS", "8. RESULTS"],
    )

    prerequisites = extract_section(
        full_text,
        "7.1 PREREQUISITE FOR THE DOCUMENTS",
        ["8. RESULTS", "9. CONSIDERATIONS"],
    )

    application = extract_section(
        full_text,
        "6. APPLICATION PROCESS",
        ["7. REQUIREMENTS", "8. RESULTS", "9. CONSIDERATIONS"],
    )

    sections = {
        "Scholarship benefits": benefits,
        "Required documents": requirements,
        "Document prerequisites": prerequisites,
        "Application process": application,
    }

    # Split the general document into overlapping chunks.
    chunks = []
    cleaned_text = re.sub(r"\s+", " ", full_text).strip()
    chunk_size = 900
    overlap = 120

    start = 0
    while start < len(cleaned_text):
        end = min(start + chunk_size, len(cleaned_text))
        chunk = cleaned_text[start:end].strip()

        if chunk:
            chunks.append({
                "text": chunk,
                "source": "General scholarship document",
            })

        if end >= len(cleaned_text):
            break

        start = end - overlap

    # Embed general chunks for semantic retrieval.
    general_embeddings = model.encode(
        [chunk["text"] for chunk in chunks]
    ).tolist()

    client = chromadb.Client()

    try:
        client.delete_collection("scholarship_chat")
    except Exception:
        pass

    collection = client.create_collection("scholarship_chat")

    collection.add(
        ids=[f"general_{i}" for i in range(len(chunks))],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=general_embeddings,
        metadatas=[{"source": chunk["source"]} for chunk in chunks],
    )

    return model, collection, sections


# =========================
# 3. Load the system
# =========================
try:
    model, collection, sections = initialize_rag()
except Exception as error:
    st.error(f"Could not initialize the RAG system: {error}")
    st.stop()


# =========================
# 4. Identify question type
# =========================
def classify_question(question):
    q = question.lower()

    if any(term in q for term in [
        "document", "documents", "requirement", "requirements",
        "passport", "transcript", "motivation letter",
        "وثائق", "الوثائق", "الأوراق", "المتطلبات",
    ]):
        return "Required documents"

    if any(term in q for term in [
        "benefit", "benefits", "allowance", "tuition",
        "insurance", "visa", "residence",
        "الفوائد", "المزايا", "البدل", "التأمين",
    ]):
        return "Scholarship benefits"

    if any(term in q for term in [
        "apply", "application process", "how to apply",
        "registration", "التقديم", "التسجيل",
    ]):
        return "Application process"

    if any(term in q for term in [
        "ielts", "toefl", "language requirement",
        "english requirement", "gpa", "minimum grade",
        "اللغة", "المعدل",
    ]):
        return "Required documents"

    return "General question"


# =========================
# 5. Retrieve context
# =========================
def retrieve_context(question):
    category = classify_question(question)

    if category == "Required documents":
        # Keep the complete list together instead of retrieving
        # only a fragment of the list.
        context_parts = []

        if sections["Required documents"]:
            context_parts.append(sections["Required documents"])

        if sections["Document prerequisites"]:
            context_parts.append(sections["Document prerequisites"])

        source_names = [
            name for name in [
                "Required documents",
                "Document prerequisites",
            ]
            if sections[name]
        ]

        return "\n\n".join(context_parts), source_names

    if category in ["Scholarship benefits", "Application process"]:
        section_text = sections.get(category, "")

        if section_text:
            return section_text, [category]

    # General questions use semantic search.
    embedding = model.encode([question]).tolist()

    result = collection.query(
        query_embeddings=embedding,
        n_results=3,
        include=["documents", "metadatas"],
    )

    docs = result["documents"][0]
    metadata = result["metadatas"][0]

    sources = list(dict.fromkeys(
        item.get("source", "General scholarship document")
        for item in metadata
    ))

    return "\n\n".join(docs), sources


# =========================
# 6. Generate an answer
# =========================
def generate_answer(question, context, category):
    instructions = {
        "Required documents": """
List every document stated in the context.
Keep documents separate from academic eligibility conditions.
Include language evidence and optional awards if stated.
Include document formatting, translation, and medical-certificate
conditions when relevant. Do not stop after the first few items.
""",
        "Scholarship benefits": """
List every scholarship benefit stated in the context.
Include tuition, flights, monthly allowance, medical insurance,
visa, and residence permit if the context mentions them.
Include the amounts and payment periods stated in the document.
""",
        "Application process": """
Explain both application stages, the responsible authorities,
the SIACAS website, the dates, and the admission channel
when those details are present in the context.
""",
        "General question": """
Answer the specific question using the context.
Include the relevant facts and figures without inventing details.
""",
    }

    prompt = f"""
You answer questions about an official scholarship document.

Follow these rules:
- Use only the information in CONTEXT.
- Answer the specific question completely.
- Do not invent facts.
- If information is missing, say so clearly.
- Do not confuse an eligibility condition with a document.

QUESTION TYPE:
{category}

SPECIAL INSTRUCTIONS:
{instructions.get(category, instructions["General question"])}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": LLM_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 450,
                "num_ctx": 4096,
            },
        },
        timeout=(10, 300),
    )

    response.raise_for_status()
    return response.json()["response"].strip()


# =========================
# 7. Chat interface
# =========================
with st.sidebar:
    st.header("About this assistant")
    st.write("Answers questions using the scholarship PDF.")
    st.write("**Embedding model:** all-MiniLM-L6-v2")
    st.write("**Language model:** Qwen 2.5 1.5B")
    st.write("**Retrieval database:** ChromaDB")

    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources used"):
                for source in message["sources"]:
                    st.write(f"- {source}")

question = st.chat_input("Ask about benefits, documents, or application...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            category = classify_question(question)
            context, sources = retrieve_context(question)

            if not context.strip():
                answer = "I could not find relevant information in the document."
            else:
                with st.spinner("Searching the document and generating an answer..."):
                    answer = generate_answer(question, context, category)

            st.markdown(answer)

            with st.expander("Sources used"):
                for source in sources:
                    st.write(f"- {source}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except requests.exceptions.Timeout:
            answer = (
                "The local AI model took too long to respond. "
                "Please try again with a shorter question."
            )
            st.warning(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

        except Exception as error:
            st.error(f"An error occurred: {error}")
