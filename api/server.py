from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI(title="RTER - Real-Time Embedding & Retrieval Scanner")

class ScanRequest(BaseModel):
    query: str
    retrieved_texts: List[str]

def dummy_embed(text: str) -> np.ndarray:
    """
    Temporary embedding function.
    Will be replaced by real embedding model.
    """
    np.random.seed(abs(hash(text)) % (10**6))
    return np.random.rand(128)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.post("/scan")
def scan(request: ScanRequest):
    query_vec = dummy_embed(request.query)
    doc_vectors = [dummy_embed(doc) for doc in request.retrieved_texts]

    similarities = [
        cosine_similarity(query_vec, doc_vec)
        for doc
