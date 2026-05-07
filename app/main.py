from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from pymupdf import pymupdf
from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient 
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise RuntimeError("HF_TOKEN is missing")

hf_client = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",
    token = hf_token
)

llm_client = InferenceClient(
    model="Qwen/Qwen2.5-3B-Instruct:featherless-ai",
    token=hf_token,
)

# model="Qwen/Qwen2.5-3B-Instruct",
# model="mistralai/Mistral-7B-Instruct-v0.3"


qdrant = QdrantClient(host="localhost", port=6333)

app = FastAPI()
router = APIRouter(prefix="/documents", tags=["documents"])

@app.get("/")
def read_root():
    return {"message": "Paper RAG OPS API"}

@router.post("/upload")
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
        
    embeddings = []

    for item in chunk_items:
        embedding = hf_client.feature_extraction(item["text"])
        embeddings.append(embedding.squeeze().tolist())
    
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
    collection_name="hf_documents",
    points=points,
    )

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content),
        "num_chunks": len(chunks)
    }


@router.post("/query")
async def query_document(question: str, top_k: int = 5):
    embedding = hf_client.feature_extraction(question)
    query_vector = embedding.squeeze().tolist()

    search_results = qdrant.query_points(
        collection_name="hf_documents",
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

    prompt = f"""
    You are a technical research assistant.

    Answer the user question using only the provided context.
    If the answer is not in the context, say you do not know.
    Cite sources using [source: chunk_id].

    Question:
    {question}

    Context:
    {context}
    """

    response = llm_client.chat_completion(
        messages=[
            {"role": "user", 
             "content": prompt
            }
        ],
        max_tokens=500,
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    return {
    "question": question,
    "answer": answer,
    "sources": retrieved_chunks,
    }

app.include_router(router)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    
    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunks.append(text[start: start+chunk_size])
        start += step

    return chunks

