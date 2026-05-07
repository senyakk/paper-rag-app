from time import struct_time

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    filename: str
    content_type: str | None
    size_bytes: int
    num_chunks: int

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default = 5, ge=1, le= 20)

class SourceChunk(BaseModel):
    chunk_id: str
    score: float
    text: str | None
    filename: str | None
    page: int | None

class Citation(BaseModel):
    chunk_id: str
    filename: str | None
    page: int | None

class Answer(BaseModel):
    answer: str | None

class QueryResponse(BaseModel):
    question: str
    result: Answer
    sources: list[SourceChunk]