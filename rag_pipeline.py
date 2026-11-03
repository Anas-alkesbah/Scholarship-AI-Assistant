
import re
from pathlib import Path

import chromadb
import pymupdf
import requests

from sentence_transformers import SentenceTransformer


class RAGPipeline:

    def __init__(
        self,
        pdf_path=None,
        embedding_name="all-MiniLM-L6-v2",
        llm_model="qwen2.5:1.5b"
    ):

        # Project configuration
        self.base_dir = Path(__file__).resolve().parent

        self.pdf_path = (
            Path(pdf_path)
            if pdf_path
            else self.base_dir / "scholarship_fixed.pdf"
        )

        self.embedding_name = embedding_name
        self.llm_model = llm_model

        # RAG components
        self.model = None
        self.collection = None
        self.sections = {}
        self.initialization_report = {}
        self.client = None

        self.collection_name = "scholarship_shared"

        self.ollama_url = "http://localhost:11434/api/generate"

    # --------------------------------
    # 1. Extract text from PDF
    # --------------------------------

    def extract_pdf_text(self):

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {self.pdf_path}"
            )

        with pymupdf.open(self.pdf_path) as pdf:

            text = "\n".join(
                page.get_text("text")
                for page in pdf
            )

        if len(text.strip()) < 100:
            raise ValueError(
                "Very little text was extracted from the PDF."
            )

        return text

    # --------------------------------
    # 2. Extract document sections
    # --------------------------------

    def extract_section(
        self,
        text,
        start_pattern,
        end_patterns
    ):

        start_match = re.search(
            start_pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE
        )

        if not start_match:
            return ""

        body_start = start_match.end()

        possible_ends = []

        for pattern in end_patterns:

            match = re.search(
                pattern,
                text[body_start:],
                flags=re.IGNORECASE | re.MULTILINE
            )

            if match:
                possible_ends.append(
                    body_start + match.start()
                )

        end = (
            min(possible_ends)
            if possible_ends
            else len(text)
        )

        return text[start_match.start():end].strip()

    # --------------------------------
    # 3. Split text into chunks
    # --------------------------------

    def split_into_chunks(
        self,
        text,
        max_words=180,
        overlap_words=30
    ):

        words = text.split()

        if not words:
            return []

        chunks = []
        start = 0

        while start < len(words):

            end = min(
                start + max_words,
                len(words)
            )

            chunks.append(
                " ".join(words[start:end])
            )

            if end >= len(words):
                break

            start = end - overlap_words

        return chunks

    # --------------------------------
    # 4. Initialize RAG system
    # --------------------------------

    def initialize(self):

        print("Extracting PDF text...")

        full_text = self.extract_pdf_text()

        section_patterns = [

            (
                "Scholarship benefits",
                r"^\s*3\s*[.]\s*SCHOLARSHIP\s+BENEFITS\b",
                [
                    r"^\s*4\s*[.]\s*TIMELINE\b",
                    r"^\s*4\s*[.]\s*DURATION\b",
                    r"^\s*5\s*[.]\s*APPLICATION\s+CRITERIA\b",
                ],
            ),

            (
                "Timeline",
                r"^\s*4\s*[.]\s*TIMELINE(?:\s+OF\s+THE\s+CALL)?\b",
                [
                    r"^\s*5\s*[.]\s*APPLICATION\s+CRITERIA\b",
                    r"^\s*6\s*[.]\s*APPLICATION\s+PROCESS\b",
                ],
            ),

            (
                "Application criteria",
                r"^\s*5\s*[.]\s*APPLICATION\s+CRITERIA\b",
                [
                    r"^\s*6\s*[.]\s*APPLICATION\s+PROCESS\b",
                    r"^\s*7\s*[.]\s*REQUIREMENTS\b",
                ],
            ),

            (
                "Application process",
                r"^\s*6\s*[.]\s*APPLICATION\s+PROCESS\b",
                [
                    r"^\s*7\s*[.]\s*REQUIREMENTS\b",
                    r"^\s*8\s*[.]\s*RESULTS\b",
                ],
            ),

            (
                "Required documents",
                r"^\s*7\s*[.]\s*REQUIREMENTS\b",
                [
                    r"^\s*7\s*[.]\s*1\s*[.]?\s*PREREQUISITE\s+FOR\s+THE\s+DOCUMENTS\b",
                    r"^\s*8\s*[.]\s*RESULTS\b",
                ],
            ),

            (
                "Document prerequisites",
                r"^\s*7\s*[.]\s*1\s*[.]?\s*PREREQUISITE\s+FOR\s+THE\s+DOCUMENTS\b",
                [
                    r"^\s*8\s*[.]\s*RESULTS\b",
                    r"^\s*9\s*[.]\s*CONSIDERATIONS\b",
                ],
            ),
        ]

        # Extract sections
        for name, start_pattern, end_patterns in section_patterns:

            self.sections[name] = self.extract_section(
                full_text,
                start_pattern,
                end_patterns
            )

        # Initialization report
        for name, content in self.sections.items():

            self.initialization_report[name] = {
                "characters": len(content),
                "preview": content[:180],
                "extracted": bool(content.strip())
            }

            print(
                f"{name}: {len(content)} characters"
            )

        # --------------------------------
        # Section-based chunking
        # --------------------------------

        chunks = []
        chunk_sections = []

        for section_name, content in self.sections.items():

            section_chunks = self.split_into_chunks(content)

            for chunk in section_chunks:

                chunks.append(chunk)
                chunk_sections.append(section_name)

        if not chunks:
            raise ValueError(
                "No chunks were created."
            )

        print(f"Number of chunks: {len(chunks)}")

        # --------------------------------
        # Load embedding model
        # --------------------------------

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            self.embedding_name
        )

        # Generate embeddings
        print("Creating embeddings...")

        embeddings = self.model.encode(
            chunks,
            show_progress_bar=False
        ).tolist()

        # --------------------------------
        # Initialize ChromaDB
        # --------------------------------

        self.client = chromadb.Client()

        try:
            self.client.delete_collection(
                self.collection_name
            )
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name
        )

        # Store chunks and metadata
        self.collection.add(
            ids=[
                f"chunk_{i}"
                for i in range(len(chunks))
            ],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {"source": section}
                for section in chunk_sections
            ]
        )

        print("RAG initialization completed!")

        return self

    # --------------------------------
    # 5. Retrieve relevant chunks
    # --------------------------------

    def retrieve(self, query, n_results=3):

        if self.model is None or self.collection is None:
            raise RuntimeError(
                "RAG pipeline is not initialized."
            )

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        query_embedding = self.model.encode(
            query
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(
                n_results,
                self.collection.count()
            )
        )

        retrieved_chunks = []

        for i, document in enumerate(results["documents"][0]):

            retrieved_chunks.append({
                "text": document,
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i]
            })

        return retrieved_chunks

    # --------------------------------
    # 6. Generate answer using Ollama
    # --------------------------------

    def generate_answer(self, query, n_results=3):

        retrieved_chunks = self.retrieve(
            query,
            n_results=n_results
        )

        if not retrieved_chunks:
            return {
                "question": query,
                "answer": "No relevant information was found.",
                "sources": []
            }

        # Build context
        context = "\n\n".join(
            f"Section: {chunk['source']}\n{chunk['text']}"
            for chunk in retrieved_chunks
        )

        # Create prompt
        prompt = f"""
You are a scholarship assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use information outside the context.
2. If the answer is not available in the context,
   clearly say that the information is not available.
3. Give clear and direct answers.
4. Do not invent facts.

Context:
{context}

Question:
{query}

Answer:
"""

        # Send request to Ollama
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        return {
            "question": query,
            "answer": result["response"].strip(),
            "sources": list(dict.fromkeys(
                chunk["source"]
                for chunk in retrieved_chunks
            ))
        }
