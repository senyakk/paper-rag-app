# Paper RAG Ops

A FastAPI-based RAG service for uploading technical documents and asking questions over them.

The app currently supports document ingestion, chunking, embeddings, Qdrant vector storage, top-k retrieval, and LLM answers with returned source chunks.

## Current Features

- Upload PDF, Markdown, and plain text files
- Parse PDF files page by page with PyMuPDF
- Chunk documents with overlapping character chunks
- Generate embeddings with Hugging Face Inference API
- Store vectors and chunk metadata in Qdrant
- Retrieve top-k relevant chunks for a question
- Generate an answer with an instruct LLM
- Return retrieved source chunks with the answer
- Auto-create the Qdrant collection on app startup
- Document the RAG design in `docs/rag_design.md`

## Tech Stack

- FastAPI
- Qdrant
- Hugging Face Inference API
- PyMuPDF
- Pydantic
- Docker Compose

## Prerequisites

Install these before running the app:

- Python 3.11+
- Docker
- Docker Compose
- A Hugging Face access token

## Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install fastapi uvicorn python-multipart pymupdf python-dotenv huggingface-hub qdrant-client
```

Create a `.env` file in the project root:

```bash
HF_TOKEN=your_hugging_face_token_here
```

Start Qdrant:

```bash
docker compose up -d
```

Start the API:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API docs are available at:

```text
http://localhost:8000/docs
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/
```

Example response:

```json
{
  "message": "Paper RAG OPS API"
}
```

### Upload A Document

Upload a PDF, Markdown, or text file:

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@path/to/your/document.pdf"
```

Example response:

```json
{
  "filename": "document.pdf",
  "content_type": "application/pdf",
  "size_bytes": 123456,
  "num_chunks": 12
}
```

### Ask A Question

```bash
curl -X POST "http://localhost:8000/documents/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "top_k": 5}'
```

Example response:

```json
{
  "question": "What is this document about?",
  "result": {
    "answer": "The document is about..."
  },
  "sources": [
    {
      "chunk_id": "d9a4f2b2-2bbf-5fd5-8b39-7a1a6a4f1e3a",
      "score": 0.82,
      "text": "...",
      "filename": "document.pdf",
      "page": 3
    }
  ]
}
```

## Project Structure

```text
.
├── app/
│   ├── main.py
│   ├── prompts.py
│   └── schemas.py
├── docs/
│   └── rag_design.md
├── add_collection.py
├── compose.yaml
└── README.md
```

## Notes

- Qdrant must be running before the FastAPI app starts.
- The app creates the `hf_documents` Qdrant collection automatically on startup.
- `add_collection.py` is now optional because collection setup happens in the app startup flow.
- Current chunking is character-based: `1000` characters with `200` characters of overlap.
- Uploading a different file with the same filename can overwrite previous chunks. A future improvement is to add a unique `document_id` per upload.

## Roadmap

- Add unique document IDs
- Add document listing and deletion endpoints
- Add structured citations
- Add an evaluation set with 50-100 question-answer pairs
- Add an evaluation runner
- Add OpenAI or Anthropic provider support
- Add screenshots of the API docs for the portfolio artifact
