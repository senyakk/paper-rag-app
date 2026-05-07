from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

"""
Manual admin script - the app doesn't depend on it
"""

collection_name = "hf_documents"

qdrant = QdrantClient(host="localhost", port=6333)
existing = qdrant.collection_exists(collection_name)

#qdrant.delete_collection(collection_name="hf_documents")


if not existing:
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size = 384,
            distance = Distance.COSINE
        ),
    )