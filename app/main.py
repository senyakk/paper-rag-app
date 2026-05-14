from fastapi import FastAPI, APIRouter, UploadFile, File
from dotenv import load_dotenv
import os

from app.loaders import read_file

from app.providers import (
    AnthropicChatProvider,
    ChatProvider,
    EmbeddingProvider,
    HFChatProvider,
    HFEmbeddingProvider,
    OpenAIChatProvider,
    OpenAIEmbeddingProvider,
)

from app.vectorstores import (
    QdrantStore,
)

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


VECTOR_SIZE = embedding_provider.dimension

vector_db = QdrantStore()

@asynccontextmanager
async def lifespan(app: FastAPI):
    vector_db.ensure_collection(VECTOR_SIZE)
    yield


app = FastAPI(lifespan=lifespan)
router = APIRouter(prefix="/documents", tags=["documents"])

@app.get("/")
def read_root():
    return {"message": "Paper RAG OPS API"}

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    
    content, chunk_items = await read_file(file)
        
    embeddings = embedding_provider.embed_texts(
        [item["text"] for item in chunk_items]
    )
    
    vector_db.upsert_chunks(chunk_items = chunk_items, embeddings=embeddings, filename=file.filename)

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

    retrieved_chunks = vector_db.search(query_vector, top_k)

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
