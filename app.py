
import streamlit as st
import requests

from rag_pipeline import RAGPipeline


# =========================
# 1. Page Configuration
# =========================

st.set_page_config(
    page_title="Scholarship AI Assistant",
    page_icon="🎓",
    layout="centered",
)


# =========================
# 2. Initialize RAG Pipeline
# =========================

@st.cache_resource(show_spinner="Loading the PDF and AI models...")
def initialize_rag():
    rag = RAGPipeline()
    rag.initialize()
    return rag


st.title("🎓 Scholarship AI Assistant")

st.caption(
    "Ask questions about the Heydar Aliyev scholarship document."
)

try:
    rag = initialize_rag()

except Exception as error:
    st.error(f"Could not initialize the RAG system: {error}")
    st.stop()


# =========================
# 3. Sidebar
# =========================

with st.sidebar:
    st.header("About this assistant")

    st.write("Answers questions using the scholarship PDF.")

    st.write(f"**Embedding model:** {rag.embedding_name}")
    st.write(f"**Language model:** {rag.llm_model}")
    st.write("**Retrieval database:** ChromaDB")

    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# =========================
# 4. Chat History
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources used"):
                for source in message["sources"]:
                    st.write(f"- {source}")


# =========================
# 5. Handle User Questions
# =========================

question = st.chat_input(
    "Ask about benefits, documents, or application..."
)


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:
            with st.spinner(
                "Searching the document and generating an answer..."
            ):
                result = rag.generate_answer(question)

            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)

            if sources:
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

        except requests.exceptions.ConnectionError:

            answer = (
                "Could not connect to Ollama. "
                "Please make sure Ollama is running."
            )

            st.error(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

        except Exception as error:

            st.error(f"An error occurred: {error}")
