from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI(title="RTER - Real-Time Embedding & Retrieval Scanner")

class ScanRequest(BaseModel):
    query: str
    retrieved_texts: List[str]


def analyze_distribution(similarities):
    if not similarities:
        return {
            "mean": 0.0,
            "std": 0.0,
            "health": "empty"
        }

    mean = float(np.mean(similarities))
    std = float(np.std(similarities))

    if std < 0.02:
        health = "collapsed"
    elif mean < 0.3:
        health = "noisy"
    else:
        health = "healthy"

    return {
        "mean": mean,
        "std": std,
        "health": health
    }

def redundancy_score(doc_vectors):
    if len(doc_vectors) < 2:
        return {"score": 0.0, "level": "low"}

    sims = []
    for i in range(len(doc_vectors)):
        for j in range(i + 1, len(doc_vectors)):
            sims.append(cosine_similarity(doc_vectors[i], doc_vectors[j]))

    avg_sim = float(np.mean(sims)) if sims else 0.0
    level = "high" if avg_sim > 0.85 else "low"

    return {
        "score": avg_sim,
        "level": level
    }

def semantic_mismatch(query_vec, doc_vectors):
    if not doc_vectors:
        return {"score": 0.0, "mismatch": False}

    centroid = np.mean(doc_vectors, axis=0)
    sim = cosine_similarity(query_vec, centroid)

    # Heuristic threshold (MVP)
    mismatch = sim < 0.5

    return {
        "score": float(sim),
        "mismatch": mismatch
    }




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
        for doc_vec in doc_vectors
    ]

    dist = analyze_distribution(similarities)
    redundancy = redundancy_score(doc_vectors)
    mismatch = semantic_mismatch(query_vec, doc_vectors)



    return {
        "status": "ok",
        "num_docs": len(request.retrieved_texts),
        "similarity_scores": similarities,
        "distribution": dist,
        "redundancy": redundancy,
        "semantic_mismatch": mismatch


    }
