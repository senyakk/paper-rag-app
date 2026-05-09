# Paper RAG Ops

A FastAPI-based RAG service for uploading technical documents and asking questions over them.

The app extracts text from uploaded files, chunks it, embeds each chunk, stores vectors in Qdrant, retrieves relevant chunks for a question, and asks an LLM to answer using the retrieved context.

## Features

- Upload PDF, Markdown, and plain text files
- Generate answers with Hugging Face, OpenAI, or Anthropic
- Retrieve top-k source chunks for each question

## Architecture

The service has four main stages:

1. Ingest an uploaded document
2. Split extracted text into chunks
3. Embed chunks and store them in Qdrant
4. Retrieve relevant chunks and generate an answer

Provider-specific code lives in `app/providers.py`. The API layer in `app/main.py` talks to providers through two small interfaces:

- `EmbeddingProvider`: exposes `dimension` and `embed_texts(...)`
- `ChatProvider`: exposes `answer(...)`

## Prerequisites

- Python 3.11+
- Docker
- Docker Compose
- API keys for the providers you choose

## Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install fastapi uvicorn python-multipart pymupdf python-dotenv huggingface-hub qdrant-client openai anthropic
```

Create a `.env` file in the project root.

Hugging Face default example:

```env
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384

LLM_PROVIDER=huggingface
LLM_MODEL=Qwen/Qwen2.5-3B-Instruct:featherless-ai

HF_TOKEN=your_hugging_face_token_here

COLLECTION_NAME=hf_documents
```

OpenAI example:

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536

LLM_PROVIDER=openai
LLM_MODEL=gpt-4.1-mini

OPENAI_API_KEY=your_openai_api_key_here

COLLECTION_NAME=openai_documents
```

Mixed provider example:

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536

LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-haiku-latest

OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

COLLECTION_NAME=openai_anthropic_documents
```

Start Qdrant:

```bash
docker compose up -d
```

Qdrant API:

```text
http://localhost:6333
```

Qdrant dashboard:

```text
http://localhost:6333/dashboard
```

Start the FastAPI app:

```bash
uvicorn app.main:app --reload
```

API:

```text
http://localhost:8000
```

Interactive API docs:

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

## Provider Configuration

Supported embedding providers:

- `huggingface`
- `openai`

Supported LLM providers:

- `huggingface`
- `openai`
- `anthropic`

The embedding provider and model determine the Qdrant vector size. If you change `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, or `EMBEDDING_DIM`, use a new `COLLECTION_NAME` or re-index the existing collection.

For example, `sentence-transformers/all-MiniLM-L6-v2` uses `384` dimensions, while `text-embedding-3-small` uses `1536` dimensions. Those vectors cannot live in the same Qdrant collection.

## Project Structure

```text
.
├── app/
│   ├── main.py
│   ├── prompts.py
│   ├── providers.py
│   └── schemas.py
├── add_collection.py
├── compose.yaml
└── README.md
```

## Notes

- The app creates the configured Qdrant collection automatically on startup.
- Current chunking is character-based: `1000` characters with `200` characters of overlap.
- Uploading a different file with the same filename can overwrite previous chunks because point IDs are based on filename, page, and chunk index.
- Anthropic is supported for answer generation, not embeddings.

## Roadmap

- Add unique document IDs
- Add document listing and deletion endpoints
- Add structured citations
- Add an evaluation set with 50-100 question-answer pairs
- Add an evaluation runner
- Add token-aware chunking
- Add screenshots of the API docs or Qdrant dashboard for the portfolio artifact
