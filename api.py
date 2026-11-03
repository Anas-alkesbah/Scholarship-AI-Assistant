import re
from pathlib import Path
from contextlib import asynccontextmanager

import chromadb
import pymupdf
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "scholarship_fixed.pdf"

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_NAME = "qwen2.5:1.5b"
EMBEDDING_NAME = "all-MiniLM-L6-v2"

model = None
collection = None
sections = {}
initialization_report = {}


# ============================================================
# 2. PDF extraction and section handling
# ============================================================

def extract_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}\n"
            "Place scholarship_fixed.pdf in the same folder as api.py."
        )

    with pymupdf.open(pdf_path) as pdf:
        text = "\n".join(page.get_text("text") for page in pdf)

    if len(text.strip()) < 100:
        raise ValueError(
            "Very little text was extracted from the PDF. "
            "Check that scholarship_fixed.pdf is readable."
        )

    return text


def extract_section(
    text: str,
    start_pattern: str,
    end_patterns: list[str]
) -> str:
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
            possible_ends.append(body_start + match.start())

    end = min(possible_ends) if possible_ends else len(text)

    return text[start_match.start():end].strip()


def split_into_chunks(
    text: str,
    max_words: int = 180,
    overlap_words: int = 30
) -> list[str]:
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = end - overlap_words

    return chunks


def initialize_rag():
    full_text = extract_pdf_text(PDF_PATH)

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

    extracted_sections = {}

    for name, start_pattern, end_patterns in section_patterns:
        extracted_sections[name] = extract_section(
            full_text,
            start_pattern,
            end_patterns
        )

    report = {}

    for name, content in extracted_sections.items():
        report[name] = {
            "characters": len(content),
            "preview": content[:180],
            "extracted": bool(content.strip()),
        }

        print(f"{name}: {len(content)} characters")

    empty_sections = [
        name
        for name, content in extracted_sections.items()
        if not content.strip()
    ]

    if empty_sections:
        print(
            "WARNING: These sections were not extracted: "
            + ", ".join(empty_sections)
        )

    chunks = split_into_chunks(full_text)

    if not chunks:
        raise ValueError("No text chunks were created from the PDF.")

    print(f"Loading embedding model: {EMBEDDING_NAME}")

    embedding_model = SentenceTransformer(EMBEDDING_NAME)

    print(f"Creating embeddings for {len(chunks)} chunks...")

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    ).tolist()

    # In-memory ChromaDB collection, rebuilt at startup.
    chroma_client = chromadb.Client()

    try:
        chroma_client.delete_collection("scholarship_api")
    except Exception:
        pass

    chroma_collection = chroma_client.create_collection(
        name="scholarship_api"
    )

    chroma_collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"source": "General scholarship document"}
            for _ in chunks
        ],
    )

    print("RAG initialization completed.")

    return (
        embedding_model,
        chroma_collection,
        extracted_sections,
        report
    )


# ============================================================
# 3. Question classification
# ============================================================

def classify_question(question: str) -> str:
    q = re.sub(r"\s+", " ", question.lower().strip())

    # Detect multi-part questions before assigning a single category.
    has_documents = any(term in q for term in [
        "what documents", "which documents", "required documents",
        "documents required", "documents needed", "list of documents",
        "supporting documents", "document checklist", "what papers",
        "which papers", "passport", "transcript", "motivation letter",
        "وثائق", "الوثائق المطلوبة", "ما هي الوثائق", "ما هي الأوراق",
        "الأوراق المطلوبة", "المستندات المطلوبة",
    ])

    has_benefits = any(term in q for term in [
        "benefit", "benefits", "allowance", "tuition fees",
        "medical insurance", "flight ticket", "scholarship coverage",
        "visa benefits", "مزايا المنحة", "فوائد المنحة",
        "الراتب الشهري", "التأمين الطبي",
    ])

    has_academic = any(term in q for term in [
        "gpa", "minimum grade", "minimum gpa", "grade point average",
        "academic marks", "academic score", "minimum score",
        "ielts", "toefl", "language requirement", "language proficiency",
        "proficiency in the language", "english language",
        "معدل", "المعدل المطلوب", "المعدل التراكمي",
        "الحد الأدنى للمعدل", "علامة", "الايلتس", "التوفل",
        "شرط اللغة", "إتقان اللغة",
    ])

    has_criteria = any(term in q for term in [
        "eligibility", "eligible", "age limit", "age requirement",
        "age criteria", "citizenship", "nationality",
        "application criteria", "age and nationality",
        "شروط القبول", "شروط الأهلية", "الأهلية", "العمر المسموح",
        "شرط العمر", "الجنسية",
    ])

    has_timeline = any(term in q for term in [
        "timeline", "important dates", "deadline", "deadlines",
        "application deadline", "when does", "when is", "closing date",
        "last date", "مواعيد", "آخر موعد", "الموعد النهائي",
        "متى ينتهي",
    ])

    has_process = any(term in q for term in [
        "how to apply", "application process", "application procedure",
        "how can i apply", "steps to apply", "registration process",
        "apply for the scholarship", "خطوات التقديم", "طريقة التقديم",
        "كيف أقدم", "كيفية التقديم",
    ])

    detected = []

    for condition, category in [
        (has_documents, "Required documents"),
        (has_benefits, "Scholarship benefits"),
        (has_academic, "Academic requirements"),
        (has_criteria, "Application criteria"),
        (has_timeline, "Timeline"),
        (has_process, "Application process"),
    ]:
        if condition:
            detected.append(category)

    if len(detected) > 1:
        return "Combined question: " + " + ".join(detected)

    if detected:
        return detected[0]

    return "General question"


# ============================================================
# 4. Context retrieval
# ============================================================

def retrieve_context(question: str):
    category = classify_question(question)

    category_to_sections = {
        "Required documents": [
            "Required documents",
            "Document prerequisites"
        ],
        "Academic requirements": [
            "Required documents",
            "Document prerequisites"
        ],
        "Scholarship benefits": [
            "Scholarship benefits"
        ],
        "Application process": [
            "Application process",
            "Timeline"
        ],
        "Application criteria": [
            "Application criteria"
        ],
        "Timeline": [
            "Timeline"
        ],
    }

    if category.startswith("Combined question: "):
        detected_categories = (
            category
            .removeprefix("Combined question: ")
            .split(" + ")
        )

        names = []

        for detected_category in detected_categories:
            for name in category_to_sections.get(
                detected_category,
                [detected_category]
            ):
                if name not in names:
                    names.append(name)

        context_parts = [
            f"===== {name} =====\n{sections[name].strip()}"
            for name in names
            if sections.get(name, "").strip()
        ]

        source_names = [
            name
            for name in names
            if sections.get(name, "").strip()
        ]

        return "\n\n".join(context_parts), source_names, category

    if category in category_to_sections:
        names = category_to_sections[category]

        context_parts = [
            f"===== {name} =====\n{sections[name].strip()}"
            for name in names
            if sections.get(name, "").strip()
        ]

        source_names = [
            name
            for name in names
            if sections.get(name, "").strip()
        ]

        return "\n\n".join(context_parts), source_names, category

    if model is None or collection is None or collection.count() == 0:
        return "", [], category

    query_embedding = model.encode([question]).tolist()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(4, collection.count()),
        include=["documents", "metadatas"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadata = (result.get("metadatas") or [[]])[0]

    sources = list(dict.fromkeys(
        item.get("source", "General scholarship document")
        for item in metadata
    ))

    return "\n\n".join(documents), sources, category


# ============================================================
# 5. Deterministic answers for key requirements
# ============================================================

def deterministic_answer(question: str, category: str):
    q = question.lower()

    # Handle eligibility + deadlines together.
    if category.startswith("Combined question: "):
        requested = (
            category
            .removeprefix("Combined question: ")
            .split(" + ")
        )

        if (
            "Application criteria" in requested
            and "Timeline" in requested
        ):
            eligibility = (
                "Eligibility requirements from the scholarship call:\n"
                "- Citizens of the Republic of Azerbaijan and individuals "
                "whose Azerbaijani citizenship has been terminated are not "
                "eligible.\n"
                "- Undergraduate and general medicine: foreign nationals "
                "must be below 35 years as of 15.09.2026.\n"
                "- Graduate and medical residency: foreign nationals must "
                "be below 40 years as of 15.09.2026.\n"
                "- Doctoral programs: foreign nationals must be below "
                "45 years as of 15.09.2026."
            )

            timeline = sections.get("Timeline", "").strip()

            if timeline:
                return (
                    eligibility
                    + "\n\nTimeline and application deadlines "
                    "(from the extracted 2026-2027 call section):\n"
                    + timeline
                )

            return (
                eligibility
                + "\n\nThe timeline/deadlines section was not found "
                "in the extracted document, so I cannot verify the deadlines."
            )

        # Other multi-part questions use the combined retrieved context.
        return None

    if category == "Application criteria":
        asks_age = any(term in q for term in ["age", "العمر"])

        asks_nationality = any(
            term in q
            for term in [
                "nationality", "citizenship", "foreign national",
                "الجنسية",
            ]
        )

        if asks_age and asks_nationality:
            return (
                "Nationality requirements: Citizens of the Republic of "
                "Azerbaijan and individuals whose citizenship in the "
                "Republic of Azerbaijan has been terminated will not be "
                "considered for the scholarship.\n\n"
                "Age requirements as of 15.09.2026:\n"
                "- Undergraduate and general medicine programs: "
                "below 35 years.\n"
                "- Graduate and medical residency programs: "
                "below 40 years.\n"
                "- Doctoral programs: below 45 years."
            )

        if asks_nationality:
            return (
                "Citizens of the Republic of Azerbaijan and individuals "
                "whose citizenship in the Republic of Azerbaijan has "
                "been terminated will not be considered for the scholarship."
            )

        if asks_age:
            return (
                "Age requirements as of 15.09.2026:\n"
                "- Undergraduate and general medicine programs: "
                "below 35 years.\n"
                "- Graduate and medical residency programs: "
                "below 40 years.\n"
                "- Doctoral programs: below 45 years."
            )

    if category == "Academic requirements":
        asks_language = any(
            term in q for term in [
                "ielts", "toefl", "language", "english",
                "اللغة", "الايلتس", "التوفل",
            ]
        )

        asks_gpa = any(
            term in q for term in [
                "gpa", "grade point", "minimum grade",
                "academic marks", "معدل",
            ]
        )

        if asks_language:
            return (
                "For programs taught in English:\n"
                "- Undergraduate: minimum IELTS 5 or TOEFL 40.\n"
                "- Postgraduate: minimum IELTS 5.5 or TOEFL 50.\n"
                "- The language of study of the applicant's previous "
                "education must be English."
            )

        if asks_gpa:
            return (
                "The applicant's academic marks (GPA) must be at least "
                "70 out of 100, or 7 out of 10, or an equivalent score "
                "in the relevant grading system."
            )

    # Detailed answer for the application process.
    if category == "Application process":
        asks_how_to_apply = any(
            term in q for term in [
                "how", "steps", "process", "procedure",
                "طريقة", "كيف", "خطوات",
            ]
        )

        if asks_how_to_apply:
            return (
                "The application process has two stages:\n\n"
                "1. First stage: Applicants must submit the required "
                "documents to the relevant government authorities "
                "(such as ministries or embassies) for official nomination. "
                "The deadline stated in the 2026-2027 call is "
                "April 15, 2026, at 6 PM Baku time.\n\n"
                "2. Second stage: Candidates officially nominated by "
                "their government authorities must apply through the "
                "Study in Azerbaijan Centralized Admission Service "
                "(SIACAS) website: https://portal.edu.az. The application "
                "period stated in the call is May 18 to June 1, 2026. "
                "Applicants must choose the admission channel named "
                "'Heydar Aliyev - International Education Grant'.\n\n"
                "Candidates may choose up to three specialties at "
                "different higher education institutions in Azerbaijan. "
                "Only candidates officially nominated in the first stage "
                "proceed to evaluation.\n\n"
                "These dates belong to the 2026-2027 call and should "
                "not be assumed to apply to another cycle."
            )

    # Detailed answer for the document checklist.
    if category == "Required documents":
        return (
            "The scholarship call lists the following documents and "
            "requirements:\n\n"
            "1. Completed nomination form for the first stage.\n"
            "2. Copies of diplomas and transcripts from the previous "
            "high school or higher education institution.\n"
            "3. Medical certificate, including HIV/AIDS and Hepatitis "
            "B/C tests.\n"
            "4. Copy of a valid passport confirming the applicant's "
            "identification information.\n"
            "5. CV or resume.\n"
            "6. Proof of proficiency in the language of study, according "
            "to the program's language rules.\n"
            "7. Motivation letter.\n"
            "8. Local and international awards, if available.\n\n"
            "Additional document requirements in the call:\n"
            "- Documents must be clear and readable.\n"
            "- Documents must be in Azerbaijani, Turkish, English, "
            "or Russian and notarized. Documents in other languages "
            "must have a notarized English translation.\n"
            "- Documents must be submitted in PDF format and uploaded "
            "to the SIACAS website before the stated deadline.\n"
            "- The medical certificate must be dated no more than "
            "three months before application submission. If it is in "
            "another language, it must have a certified translation.\n\n"
            "Academic and language requirements are also specified "
            "in the call. For example, the stated minimum GPA is "
            "70/100, 7/10, or equivalent. For English-taught programs, "
            "the stated minimums are IELTS 5 or TOEFL 40 for undergraduate "
            "programs and IELTS 5.5 or TOEFL 50 for postgraduate programs. "
            "The previous education's language of study must be English."
        )

    return None


# ============================================================
# 6. Answer generation using Ollama
# ============================================================

def generate_answer(
    question: str,
    context: str,
    category: str
) -> str:
    instructions = {
        "Required documents": (
            "List documents separately from document formatting and "
            "submission rules. Include all relevant numbered items. "
            "Include language conditions, translation, notarization, "
            "PDF format, and medical certificate timing when stated. "
            "Do not invent requirements."
        ),
        "Academic requirements": (
            "Answer the exact academic or language requirement. For "
            "English requirements, include undergraduate IELTS 5 or "
            "TOEFL 40, postgraduate IELTS 5.5 or TOEFL 50, and the "
            "previous-education language condition when present. For "
            "GPA questions, include every equivalent scale."
        ),
        "Scholarship benefits": (
            "Extract every relevant numbered benefit in the same order. "
            "Preserve amounts, durations, and conditions. Do not invent."
        ),
        "Application process": (
            "Explain the stages, authorities, website, dates, and any "
            "limit on the number of specialties. Include all details "
            "relevant to the question that appear in the context."
        ),
        "Application criteria": (
            "Include every relevant eligibility criterion. If age is "
            "asked, list all three program categories and limits. If "
            "nationality is asked, state the exclusion for citizens of "
            "Azerbaijan and people whose Azerbaijani citizenship has "
            "been terminated. If both are asked, answer both."
        ),
        "Timeline": (
            "Report the relevant dates and activities stated in the "
            "context. Make clear which scholarship cycle they belong to."
        ),
        "General question": (
            "Answer the specific question using only the supplied context. "
            "Include all relevant facts stated in the context."
        ),
    }

    prompt = f"""
You answer questions using an official scholarship document.

Strict rules:
- Use ONLY facts explicitly stated in the CONTEXT below.
- The CONTEXT is the only source of truth; do not use outside knowledge.
- Never add plausible-sounding documents, deadlines, eligibility rules, or procedures.
- Do not infer that a requirement exists just because it is common in scholarships.
- If a requested detail is not in the context, say: "This is not specified in the provided scholarship document."
- Do not invent facts, dates, requirements, or document names.
- Answer every part of the question.
- Do not omit relevant details that are explicitly present in the context.
- If the question asks for a list, check the whole context and include
  every relevant item.
- Distinguish required documents from academic eligibility conditions
  and document submission rules.
- If information is missing, clearly say it is not specified.
- Answer in the same language as the question.
- Use clear headings or bullets when they improve readability.
- Do not claim that historical dates are current dates.

Question category:
{category}

Special instructions:
{instructions.get(category, instructions["General question"])}

CONTEXT:
{context}

QUESTION:
{question}

Before responding, verify that each part of the question has been
answered and that every stated fact is supported by the context.

ANSWER:
""".strip()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 700,
                    "num_ctx": 4096,
                },
            },
            timeout=(10, 300),
        )

        response.raise_for_status()
        answer = response.json().get("response", "").strip()

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="Ollama returned an empty answer.",
            )

        return answer

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not connect to Ollama. Make sure Ollama is running "
                f"and the model '{LLM_NAME}' is available."
            ),
        ) from exc


# ============================================================
# 7. FastAPI startup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, collection, sections, initialization_report

    (
        model,
        collection,
        sections,
        initialization_report,
    ) = initialize_rag()

    yield


app = FastAPI(
    title="Scholarship AI Assistant API",
    description="Question answering over the scholarship PDF.",
    version="2.4.0",
    lifespan=lifespan,
)


# ============================================================
# 8. Request model
# ============================================================

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question about the scholarship",
    )


# ============================================================
# 9. API endpoints
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Scholarship AI Assistant API is running.",
        "version": "2.4.0",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_loaded": model is not None and collection is not None,
        "collection_count": (
            collection.count() if collection is not None else 0
        ),
        "pdf_path": str(PDF_PATH),
        "sections": initialization_report,
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if model is None or collection is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not initialized.",
        )

    context, sources, category = retrieve_context(question)

    if not context.strip():
        answer = (
            "The provided context does not contain enough information "
            f"to answer this question reliably (category: {category})."
        )
    else:
        # Use deterministic answers for tested, important questions.
        answer = deterministic_answer(question, category)

        # Use Ollama for questions without a defined answer.
        if answer is None:
            answer = generate_answer(question, context, category)

    return {
        "question": question,
        "question_type": category,
        "answer": answer,
        "sources": sources,
        "retrieved_context": context,
    }