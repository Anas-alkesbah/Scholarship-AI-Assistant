# 🎓 Scholarship AI Assistant

An AI-powered question-answering system that helps applicants understand the **Heydar Aliyev International Education Grant Program (2026–2027)**.

The project uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from an official scholarship PDF and generate answers using a locally running Large Language Model (LLM).

## ✨ Features

- **PDF Processing:** Extracts and processes scholarship information from PDF documents.
- **Semantic Search:** Uses sentence embeddings to retrieve relevant information.
- **Vector Database:** Stores document embeddings using ChromaDB.
- **AI-Powered Answers:** Generates context-based answers using Qwen 2.5.
- **Source References:** Displays the document sections used to generate answers.
- **Interactive Chat Interface:** Provides a conversational interface built with Streamlit.
- **REST API:** Exposes question-answering functionality through FastAPI.
- **Local LLM Execution:** Uses Ollama to run the language model locally.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web interface |
| FastAPI | REST API development |
| Sentence Transformers | Text embeddings |
| ChromaDB | Vector database |
| Ollama | Local LLM execution |
| Qwen 2.5 (1.5B) | Language model |
| PyMuPDF | PDF text extraction |

## 🧠 RAG Architecture

The system follows a Retrieval-Augmented Generation (RAG) workflow:

1. **Document Loading:** Extracts text from the scholarship PDF using PyMuPDF.
2. **Text Processing:** Splits the extracted text into smaller overlapping chunks.
3. **Embedding Generation:** Converts text chunks into vector representations using `all-MiniLM-L6-v2`.
4. **Vector Storage:** Stores embeddings in ChromaDB.
5. **Information Retrieval:** Retrieves relevant chunks based on the user's question.
6. **Answer Generation:** Sends the retrieved context and question to Qwen 2.5 through Ollama.
7. **Response:** Returns an answer with its corresponding source sections.

## 📁 Project Structure

```text
Scholarship-AI-Assistant/
│
├── app.py                  # Streamlit chat interface
├── api.py                  # FastAPI application
├── rag_pipeline.py         # Core RAG implementation
├── scholarship_fixed.pdf   # Scholarship document
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Anas-alkesbah/Scholarship-AI-Assistant.git
cd Scholarship-AI-Assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Configure Ollama

Download and install Ollama:

https://ollama.com/download

Pull the required language model:

```bash
ollama pull qwen2.5:1.5b
```

Make sure Ollama is running before starting the application.

## 🚀 Running the Application

### Streamlit Interface

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

### FastAPI Server

```bash
uvicorn api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## 🔌 API Usage

### POST `/ask`

Send a question about the scholarship.

**Request:**

```json
{
  "question": "What is the minimum GPA required?"
}
```

**Response:**

```json
{
  "question": "What is the minimum GPA required?",
  "answer": "The minimum GPA required is 70...",
  "sources": [
    "Required documents",
    "Application criteria"
  ]
}
```

### GET `/health`

Checks whether the RAG system has been initialized and returns information about the loaded document and vector collection.

## 📌 Current Scope

The current implementation is designed to answer questions based on the provided scholarship document.

The system uses local model execution through Ollama and requires the PDF document to be available in the project directory.

## 👨‍💻 Author

**Anas Alkesbah**

Artificial Intelligence Graduate

[GitHub](https://github.com/Anas-alkesbah)
