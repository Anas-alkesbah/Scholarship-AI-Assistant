# 🎓 Scholarship AI Assistant

An AI-powered assistant that answers questions about the Heydar Aliyev International Education Grant Program using information extracted from the official scholarship document.

The application combines **Retrieval-Augmented Generation (RAG)** with a local Large Language Model (LLM) to help users find information about scholarship benefits, required documents, and application procedures.

## ✨ Features

- **PDF Processing:** Extracts text from the scholarship document.
- **Semantic Search:** Uses Sentence Transformers to represent text as embeddings.
- **Vector Database:** Uses ChromaDB to retrieve relevant information.
- **AI-Powered Answers:** Uses Qwen 2.5 1.5B through Ollama to generate answers locally.
- **Interactive Interface:** Built with Streamlit.
- **Source References:** Displays the document sections used to answer questions.

## 🛠️ Technologies Used

- Python
- Streamlit
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- ChromaDB
- Ollama
- Qwen 2.5 1.5B
- PyMuPDF

## 📁 Project Structure

```text
Scholarship-AI-Assistant/
├── app.py
├── scholarship_fixed.pdf
├── requirements.txt
└── README.md
```

## ⚙️ Requirements

Before running the application, install:

- Python
- Ollama
- The Qwen 2.5 1.5B model
- The Python dependencies listed in `requirements.txt`

## 🚀 Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Anas-alkesbah/Scholarship-AI-Assistant.git
cd Scholarship-AI-Assistant
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and start the local language model

Install Ollama from [ollama.com](https://ollama.com/), then download the model:

```bash
ollama pull qwen2.5:1.5b
```

Make sure the Ollama service is running before launching the application.

### 4. Run the application

Ensure that `scholarship_fixed.pdf` is in the same directory as `app.py`, then run:

```bash
streamlit run app.py
```

The application will open in your browser, usually at `http://localhost:8501`.

## 💡 Example Questions

- What are the scholarship benefits?
- What documents are required for the scholarship?
- How do applicants apply for the scholarship?
- What language proficiency requirements apply?

## 🧠 How It Works

1. Extracts text from the scholarship PDF.
2. Splits the document into overlapping text chunks.
3. Converts text into embeddings using Sentence Transformers.
4. Stores the embeddings in ChromaDB.
5. Retrieves relevant information based on the user's question.
6. Sends the retrieved context to the Qwen language model to generate an answer.

## ⚠️ Notes and Limitations

- The application requires a local Ollama installation and a downloaded language model.
- Answer quality depends on the extracted document text and the language model's ability to use the retrieved context.
- Scholarship deadlines and requirements may change. Always consult the latest official scholarship announcement before applying.
- This project is an educational AI application and is not an official service of the scholarship provider.

## 👨‍💻 Author

**Anas Al-Kesbah**

AI Graduate | Artificial Intelligence Projects

---

Built as a practical project exploring RAG, local language models, semantic search, and AI application development.
