# RAG Design

## Overview

This app is a document ingestion and retrieval-augmented generation service. Users upload PDF, Markdown, or text files. The app extracts text, chunks it, embeds each chunk, stores the vectors in Qdrant, retrieves relevant chunks for a question, and asks an LLM to answer using only the retrieved context.

## Architecture

The service has four main stages:

1. Document ingestion
2. Chunking and embedding
3. Vector storage and retrieval
4. Answer generation with source chunks

## Ingestion Flow

Users upload documents through the `POST /documents/upload` endpoint.

Supported file types:

- PDF
- Markdown
- Plain text

PDF files are parsed page by page with PyMuPDF. Markdown and text files are decoded as UTF-8.

The current implementation stores chunks directly in Qdrant. Point IDs are generated from filename, page, and chunk index, so uploading a different file with the same filename can overwrite existing chunks. A future improvement is to add a unique `document_id` per upload.

## Chunking Strategy

Text is split into overlapping character chunks.

Current settings:

- Chunk size: `1000` characters
- Overlap: `200` characters

Overlap helps preserve context across chunk boundaries.

Each chunk stores this metadata:

- `filename`
- `page`
- `text`

For text and Markdown files, `page` is stored as `null`.

## Embeddings

Chunks are embedded with:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding size:

```text
384 dimensions
```

The embedding model is served through the Hugging Face Inference API.

## Vector Store

The app uses Qdrant as the vector database.

Collection name:

```text
hf_documents
```

Distance metric:

```text
Cosine similarity
```

The app automatically creates the Qdrant collection on startup if it does not already exist.

## Retrieval Flow

Users ask questions through the `POST /documents/query` endpoint.

The question is embedded with the same embedding model used for documents. The app retrieves the top-k most similar chunks from Qdrant.

Default retrieval setting:

```text
top_k = 5
```

Each retrieved source includes:

- Chunk ID
- Similarity score
- Filename
- Page
- Text

## Generation Flow

Retrieved chunks are formatted into a context block and passed to the LLM.

The prompt instructs the model to:

- Answer only using the provided context
- Say it does not know when the answer is missing
- Cite sources using chunk IDs

The app currently uses:

```text
Qwen/Qwen2.5-3B-Instruct:featherless-ai
```

The LLM is called through the Hugging Face Inference API.

## API Endpoints

### `GET /`

Health check endpoint.

### `POST /documents/upload`

Uploads and indexes a document.

Returns:

- `filename`
- `content_type`
- `size_bytes`
- `num_chunks`

Example response:

```json
{
  "filename": "paper.pdf",
  "content_type": "application/pdf",
  "size_bytes": 123456,
  "num_chunks": 12
}
```

### `POST /documents/query`

Answers a question over indexed documents.

Example request body:

```json
{
  "question": "What is this document about?",
  "top_k": 5
}
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
      "filename": "paper.pdf",
      "page": 3
    }
  ]
}
```

## Citation Design

The prompt asks the LLM to cite sources using chunk IDs, and the API response returns the retrieved source chunks so a user can inspect where the answer came from.

Current source metadata:

- `chunk_id`
- `score`
- `text`
- `filename`
- `page`

Future improvements:

- Add a structured `citations` field to the response model
- Require the LLM to return citations in a structured format
- Validate that cited chunk IDs exist in the retrieved sources
- Display human-readable citations like `paper.pdf, page 3`

## Limitations

Current limitations:

- Character-based chunking instead of semantic or token-aware chunking
- Duplicate filenames can overwrite previous chunks
- No document-level ID in the upload response
- No document listing endpoint
- No document deletion endpoint
- No user authentication
- No reranking step
- No evaluation runner yet
- No streaming responses
- No UI yet
- LLM citations are prompt-guided but not strictly validated

## Future Improvements

Planned improvements:

- Add unique `document_id` values for uploads
- Add token-aware chunking
- Add document listing and deletion
- Add evaluation set with 50-100 question-answer pairs
- Add automated retrieval and answer quality evaluation
- Add structured LLM output with validated citations
- Add OpenAI or Anthropic provider support
- Add a simple frontend or API docs screenshots for portfolio presentation
