from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from pymupdf import pymupdf
from dotenv import load_dotenv
import os

from app.providers import (
    AnthropicChatProvider,
    ChatProvider,
    EmbeddingProvider,
    HFChatProvider,
    HFEmbeddingProvider,
    OpenAIChatProvider,
    OpenAIEmbeddingProvider,
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

from contextlib import asynccontextmanager
from app.prompts import build_rag_prompt

from app.schemas import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    Answer
)


load_dotenv()

embedding_provider: EmbeddingProvider
llm_provider: ChatProvider

embedding_provider_name = os.getenv("EMBEDDING_PROVIDER", "huggingface")
llm_provider_name = os.getenv("LLM_PROVIDER", "huggingface")

match embedding_provider_name:
    case "huggingface":
        embedding_provider = HFEmbeddingProvider()
    case "openai":
        embedding_provider = OpenAIEmbeddingProvider()
    case _:
        raise RuntimeError(f"Unsupported EMBEDDING_PROVIDER: {embedding_provider_name}")

match llm_provider_name:
    case "huggingface":
        llm_provider = HFChatProvider()
    case "openai":
        llm_provider = OpenAIChatProvider()
    case "anthropic":
        llm_provider = AnthropicChatProvider()
    case _:
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {llm_provider_name}")


COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hf_documents")
VECTOR_SIZE = embedding_provider.dimension

qdrant = QdrantClient(host="localhost", port=6333)


def get_collection_vector_size(collection_name: str) -> int | None:
    collection = qdrant.get_collection(collection_name=collection_name)
    vectors = collection.config.params.vectors

    if isinstance(vectors, dict):
        first_vector = next(iter(vectors.values()), None)
        return getattr(first_vector, "size", None)

    return getattr(vectors, "size", None)


def check_collection():
    if qdrant.collection_exists(collection_name=COLLECTION_NAME):
        existing_size = get_collection_vector_size(COLLECTION_NAME)
        if existing_size is not None and existing_size != VECTOR_SIZE:
            raise RuntimeError(
                f"Qdrant collection '{COLLECTION_NAME}' uses vector size "
                f"{existing_size}, but the configured embedding provider uses "
                f"{VECTOR_SIZE}. Use a different COLLECTION_NAME or re-index "
                "the collection."
            )
        return
    
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config = VectorParams(
            size = VECTOR_SIZE,
            distance = Distance.COSINE
        )
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    check_collection()
    yield


app = FastAPI(lifespan=lifespan)
router = APIRouter(prefix="/documents", tags=["documents"])

@app.get("/")
def read_root():
    return {"message": "Paper RAG OPS API"}

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):

    allowed_extensions = [".txt", ".md", ".pdf"]
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")

    if not any(file.filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .md, and .pdf files are supported"
        )
    
    content = await file.read()

    if file.filename.endswith(".pdf"):

        pdf = pymupdf.open(stream=content, filetype="pdf")
        
        chunk_items = []

        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text()
            chunks = chunk_text(page_text, chunk_size=1000, overlap=200)

            for chunk in chunks:
                chunk_items.append({
                    "text": chunk,
                    "page": page_number,
                })

    else:
        try:
            text = content.decode("utf-8")
            chunks = chunk_text(text, chunk_size = 1000, overlap = 200)

            chunk_items = []

            for chunk in chunks:
                chunk_items.append({
                    "text": chunk,
                    "page": None,
                })

        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Could not decode file as UTF-8 text"
            )
        
    embeddings = embedding_provider.embed_texts(
        [item["text"] for item in chunk_items]
    )
    
    points = []

    for chunk_index, (item, embedding) in enumerate(zip(chunk_items, embeddings)):
        stable_key = f"{file.filename}:{item['page']}:{chunk_index}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": item["text"],
                    "filename": file.filename,
                    "page": item["page"],
                },
            )
        )

    qdrant.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    )

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content),
        "num_chunks": len(chunk_items)
    }


@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    question = request.question
    top_k = request.top_k
    query_vector = embedding_provider.embed_texts([question])[0]

    search_results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query = query_vector,
        limit=top_k,
        with_payload=True
    )

    retrieved_chunks = []

    for point in search_results.points:
        retrieved_chunks.append({
            "chunk_id": str(point.id),
            "score": point.score,
            "text": point.payload.get("text"),
            "filename": point.payload.get("filename"),
            "page": point.payload.get("page"),
        })

    context = "\n\n".join(
        f"[source: {chunk['chunk_id']}]\n"
        f"Filename: {chunk['filename']}\n"
        f"Page: {chunk['page']}\n"
        f"{chunk['text']}"
        for chunk in retrieved_chunks
    )

    prompt = build_rag_prompt(question = question, context=context)
    answer = llm_provider.answer(prompt)

    sources = [
        SourceChunk(**chunk)
        for chunk in retrieved_chunks
    ]

    return QueryResponse(
        question=question,
        result = Answer(
            answer = answer
        ),
        sources=sources
    )

app.include_router(router)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    
    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunks.append(text[start: start+chunk_size])
        start += step

    return chunks
