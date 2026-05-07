
def build_rag_prompt(question: str, context: str) -> str:
    return f"""
    You are a technical research assistant.

    Answer the user question using only the provided context.
    If the answer is not in the context, say you do not know.
    Cite sources using [source: chunk_id].

    Question:
    {question}

    Context:
    {context}
    """.strip()