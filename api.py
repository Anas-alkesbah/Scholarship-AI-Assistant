
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_pipeline import RAGPipeline


# =========================
# 1. Initialize RAG Pipeline
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Initializing Scholarship AI Assistant API...")

    try:
        rag = RAGPipeline()
        rag.initialize()

        app.state.rag = rag

        print("API is ready!")

    except Exception as error:
        print(f"RAG initialization failed: {error}")
        raise

    yield

    print("API shutting down...")


# =========================
# 2. FastAPI Configuration
# =========================

app = FastAPI(
    title="Scholarship AI Assistant API",
    description="Question answering over the Heydar Aliyev scholarship PDF using RAG.",
    version="3.0.0",
    lifespan=lifespan,
)


# =========================
# 3. Request Model
# =========================

class QuestionRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="Question about the scholarship",
    )


# =========================
# 4. API Endpoints
# =========================

@app.get("/")
def home():

    return {
        "message": "Scholarship AI Assistant API is running.",
        "version": "3.0.0",
    }


@app.get("/health")
def health():

    rag = getattr(app.state, "rag", None)

    if rag is None:
        return {
            "status": "not ready",
            "rag_loaded": False,
        }

    return {
        "status": "ok",
        "rag_loaded": True,
        "collection_count": rag.collection.count(),
        "pdf_path": str(rag.pdf_path),
        "sections": rag.initialization_report,
        "embedding_model": rag.embedding_name,
        "llm_model": rag.llm_model,
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    rag = getattr(app.state, "rag", None)

    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not initialized.",
        )

    try:

        result = rag.generate_answer(question)

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except requests.exceptions.Timeout:

        raise HTTPException(
            status_code=504,
            detail="The AI model took too long to respond.",
        )

    except requests.exceptions.ConnectionError:

        raise HTTPException(
            status_code=503,
            detail="Could not connect to Ollama. Make sure Ollama is running.",
        )

    except requests.exceptions.RequestException as error:

        raise HTTPException(
            status_code=502,
            detail=f"Error communicating with Ollama: {str(error)}",
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(error)}",
        )
