from pymupdf import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import UploadFile, File, HTTPException


text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200
            )


async def read_file(file: UploadFile = File(...)) -> list:

    allowed_extensions = [".txt", ".md", ".pdf"]
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename")

    if not any(file.filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .md, and .pdf files are supported"
        )


    content = await file.read()
    chunk_items = []

    if file.filename.endswith(".pdf"):

        pdf = pymupdf.open(stream=content, filetype="pdf")
        
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text()

            chunks = text_splitter.split_text(page_text)

            for chunk in chunks:
                chunk_items.append({
                    "text": chunk,
                    "page": page_number,
                })

    else:
        try:
            text = content.decode("utf-8")
            chunks = text_splitter.split_text(text)

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
        
    return [content, chunk_items]