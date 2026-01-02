from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import os

# Thresholds (env-overridable)
SIM_STD_COLLAPSE = float(os.getenv("SIM_STD_COLLAPSE", 0.02))
MEAN_NOISY = float(os.getenv("MEAN_NOISY", 0.3))
REDUNDANCY_HIGH = float(os.getenv("REDUNDANCY_HIGH", 0.85))
MISMATCH_SIM = float(os.getenv("MISMATCH_SIM", 0.5))
DRIFT_TOL_MEAN = float(os.getenv("DRIFT_TOL_MEAN", 0.1))
DRIFT_TOL_STD = float(os.getenv("DRIFT_TOL_STD", 0.05))

app = FastAPI(title="RTER - Real-Time Embedding & Retrieval Scanner")

class ScanRequest(BaseModel):
    query: str
    retrieved_texts: List[str]
    baseline: dict | None = None


def analyze_distribution(similarities):
    if not similarities:
        return {
            "mean": 0.0,
            "std": 0.0,
            "health": "empty"
        }

    mean = float(np.mean(similarities))
    std = float(np.std(similarities))

    if std < SIM_STD_COLLAPSE:
        health = "collapsed"
    elif mean < MEAN_NOISY:
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
    level = "high" if avg_sim > REDUNDANCY_HIGH else "low"


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
    mismatch = sim < MISMATCH_SIM


    return {
        "score": float(sim),
        "mismatch": mismatch
    }

def decision_engine(distribution, redundancy, mismatch):
    reasons = []

    if mismatch["mismatch"]:
        reasons.append("semantic mismatch detected")

    if redundancy["level"] == "high":
        reasons.append("high redundancy in retrieved documents")

    if distribution["health"] == "collapsed":
        reasons.append("similarity distribution collapsed")

    if reasons:
        status = "error" if mismatch["mismatch"] else "warn"
    else:
        status = "ok"

    return {
        "status": status,
        "reasons": reasons
    }

def drift_check(current_dist, baseline):
    if not baseline:
        return {"drift": False, "reason": "no baseline"}

    mean_shift = abs(current_dist["mean"] - baseline["mean"])
    std_shift = abs(current_dist["std"] - baseline["std"])

    drift = (mean_shift > DRIFT_TOL_MEAN) or (std_shift > DRIFT_TOL_STD)
    reasons = []
    if mean_shift > DRIFT_TOL_MEAN:
        reasons.append("mean similarity shifted")
    if std_shift > DRIFT_TOL_STD:
        reasons.append("similarity variance shifted")

    return {
        "drift": drift,
        "mean_shift": mean_shift,
        "std_shift": std_shift,
        "reasons": reasons
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
    drift = drift_check(dist, request.baseline)
    redundancy = redundancy_score(doc_vectors)
    mismatch = semantic_mismatch(query_vec, doc_vectors)
    decision = decision_engine(dist, redundancy, mismatch)



    return {
        "status": "ok",
        "num_docs": len(request.retrieved_texts),
        "similarity_scores": similarities,
        "distribution": dist,
        "redundancy": redundancy,
        "semantic_mismatch": mismatch,
        "decision": decision,
        "drift": drift




    }
