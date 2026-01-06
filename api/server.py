from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from typing import Literal
import numpy as np
import os,re
from enum import Enum
from collections import deque
from typing import Optional

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response,HTTPException

import time


REQ_COUNTER = Counter(
    "rter_requests_total",
    "Total RTER scan requests",
    ["status", "severity"]
)

DRIFT_COUNTER = Counter(
    "rter_drift_total",
    "Total drift detections"
)

LATENCY = Histogram(
    "rter_scan_latency_seconds",
    "Latency of /scan endpoint"
)


BASELINE_WINDOW = int(os.getenv("BASELINE_WINDOW", 50))
baseline_buffer = deque(maxlen=BASELINE_WINDOW)


# Thresholds (env-overridable)
SIM_STD_COLLAPSE = float(os.getenv("SIM_STD_COLLAPSE", 0.02))
MEAN_NOISY = float(os.getenv("MEAN_NOISY", 0.3))
REDUNDANCY_HIGH = float(os.getenv("REDUNDANCY_HIGH", 0.85))
MISMATCH_SIM = float(os.getenv("MISMATCH_SIM", 0.5))
DRIFT_TOL_MEAN = float(os.getenv("DRIFT_TOL_MEAN", 0.1))
DRIFT_TOL_STD = float(os.getenv("DRIFT_TOL_STD", 0.05))


app = FastAPI(title="RTER - Real-Time Embedding & Retrieval Scanner")

class Severity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


class ReasonCode(str, Enum):
    SEMANTIC_MISMATCH = "SEMANTIC_MISMATCH"
    HIGH_REDUNDANCY = "HIGH_REDUNDANCY"
    COLLAPSED_DISTRIBUTION = "COLLAPSED_DISTRIBUTION"
    NOISY_DISTRIBUTION = "NOISY_DISTRIBUTION"
    DRIFT_MEAN_SHIFT = "DRIFT_MEAN_SHIFT"
    DRIFT_STD_SHIFT = "DRIFT_STD_SHIFT"


class ScanRequest(BaseModel):
    # Text inputs (optional)
    query: Optional[str] = None
    retrieved_texts: Optional[List[str]] = None

    # Embeddings (optional)
    query_embedding: Optional[List[float]] = None
    doc_embeddings: Optional[List[List[float]]] = None

    baseline: Optional[dict] = None
    mode: Literal["retrieval", "rewrite"] = "retrieval"

class BatchScanRequest(BaseModel):
    items: List[ScanRequest]

class BatchScanResponse(BaseModel):
    results: List[dict]
    summary: dict

def keyword_overlap_ratio(a: str, b: str) -> float:
    tokenize = lambda s: set(re.findall(r"[a-z]+", s.lower()))
    ta, tb = tokenize(a), tokenize(b)

    if not ta:
        return 0.0

    return len(ta & tb) / len(ta)
def severity_from_decision(decision):
    codes = set(decision.get("reason_codes", []))
    if "SEMANTIC_MISMATCH" in codes:
        return Severity.CRITICAL
    if any(c in codes for c in ["HIGH_REDUNDANCY", "COLLAPSED_DISTRIBUTION", "DRIFT_MEAN_SHIFT", "DRIFT_STD_SHIFT"]):
        return Severity.WARN
    return Severity.INFO

def suggested_actions_from_reasons(reason_codes: list[str]):
    print("DEBUG reason_codes:", reason_codes)
    actions = set()

    for code in reason_codes:
        code = code.strip().upper()  # 👈 NORMALIZE

        if code == "COLLAPSED_DISTRIBUTION":
            actions.update([
                "increase_top_k",
                "use_hybrid_retrieval",
                "add_keyword_filter"
            ])

        elif code == "HIGH_REDUNDANCY":
            actions.update([
                "deduplicate_documents",
                "increase_retrieval_diversity"
            ])

        elif code == "SEMANTIC_MISMATCH":
            actions.update([
                "fallback_to_keyword_search",
                "block_generation",
                "request_more_context"
            ])

        elif code == "MEANING_DRIFT_HIGH":
            actions.update([
                "regenerate_output",
                "lower_temperature",
                "compare_with_original_text"
            ])

        elif code == "MEANING_DRIFT_MEDIUM":
            actions.update([
                "log_warning",
                "manual_review"
            ])

    return list(actions)




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

def semantic_mismatch(query: str, doc_texts: list[str], query_vec, doc_vecs):
    sims = [cosine_similarity(query_vec, dv) for dv in doc_vecs]
    max_sim = max(sims)
    mean_sim = sum(sims) / len(sims)

    overlaps = [keyword_overlap_ratio(query, t) for t in doc_texts]
    max_overlap = max(overlaps) if overlaps else 0.0

    mismatch = (
        max_sim < 0.70 or        # weak semantic similarity
        mean_sim < 0.72 or
        max_overlap < 0.15       # 🔥 topic anchor guard
    )

    return {
        "score": round(max_sim, 3),
        "mismatch": mismatch,
        "lexical_overlap": round(max_overlap, 3)
    }


def decision_engine(distribution, redundancy, mismatch, drift=None):
    reasons = []
    codes = []

    if mismatch["mismatch"]:
        reasons.append("semantic mismatch detected")
        codes.append(ReasonCode.SEMANTIC_MISMATCH)

    if redundancy["level"] == "high":
        reasons.append("high redundancy in retrieved documents")
        codes.append(ReasonCode.HIGH_REDUNDANCY)

    if distribution["health"] == "collapsed":
        reasons.append("similarity distribution collapsed")
        codes.append(ReasonCode.COLLAPSED_DISTRIBUTION)

    if distribution["health"] == "noisy":
        reasons.append("noisy similarity distribution")
        codes.append(ReasonCode.NOISY_DISTRIBUTION)

    if drift and drift.get("drift"):
        for r in drift.get("reasons", []):
            if "mean" in r:
                codes.append(ReasonCode.DRIFT_MEAN_SHIFT)
            if "variance" in r:
                codes.append(ReasonCode.DRIFT_STD_SHIFT)

    if codes:
        status = "error" if ReasonCode.SEMANTIC_MISMATCH in codes else "warn"
    else:
        status = "ok"

    return {
        "status": status,
        "reasons": reasons,
        "reason_codes": [c.value for c in codes]
    }


def drift_check(current_dist, baseline):
    if not baseline:
        return {
            "drift": False,
            "reason": "no baseline"
        }

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

def update_and_get_baseline(dist, decision):
    if decision is None:
        return None

    # learn only from healthy decisions
    if decision["status"] == "ok" and dist["health"] == "healthy":
        baseline_buffer.append({
            "mean": dist["mean"],
            "std": dist["std"]
        })

    if not baseline_buffer:
        return None

    mean_vals = [b["mean"] for b in baseline_buffer]
    std_vals = [b["std"] for b in baseline_buffer]

    return {
        "mean": float(np.mean(mean_vals)),
        "std": float(np.mean(std_vals))
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

def resolve_embeddings(request: ScanRequest):
    # Case 1: User provided embeddings
    if request.query_embedding and request.doc_embeddings:
        return (
            np.array(request.query_embedding),
            [np.array(e) for e in request.doc_embeddings]
        )
    

    # Case 2: Fallback to text-based embedding
    if request.query and request.retrieved_texts:
        query_vec = dummy_embed(request.query)
        doc_vecs = [dummy_embed(t) for t in request.retrieved_texts]
        return query_vec, doc_vecs

    raise ValueError("Either embeddings or text inputs must be provided")



def validate_embeddings(query_vec: np.ndarray, doc_vecs: list[np.ndarray]):
    if query_vec.size == 0:
        raise HTTPException(status_code=400, detail="query_embedding is empty")

    dims = query_vec.shape[0]

    for i, v in enumerate(doc_vecs):
        if v.size == 0:
            raise HTTPException(status_code=400, detail=f"doc_embeddings[{i}] is empty")
        if v.shape[0] != dims:
            raise HTTPException(
                status_code=400,
                detail=f"Embedding dimension mismatch: query={dims}, doc[{i}]={v.shape[0]}"
            )
        if not np.isfinite(v).all():
            raise HTTPException(
                status_code=400,
                detail=f"Non-finite values in doc_embeddings[{i}]"
            )

    if not np.isfinite(query_vec).all():
        raise HTTPException(
            status_code=400,
            detail="Non-finite values in query_embedding"
        )
def meaning_drift_score(a: np.ndarray, b: np.ndarray):
    sim = cosine_similarity(a, b)
    drift = float(max(0.0, min(1.0, 1.0 - sim)))

    if drift <= 0.2:
        level = "low"
    elif drift <= 0.4:
        level = "medium"
    else:
        level = "high"

    return {
        "similarity": float(sim),
        "score": drift,
        "level": level
    }
def confidence_score(dist, redundancy, mismatch, meaning_drift, drift):
    risk = 0.0

    # Distribution collapse
    if dist.get("health") == "collapsed" and dist.get("std", 0) < 0.015:
        risk += 0.15


    # Redundancy (0–1)
    risk += min(0.15, float(redundancy.get("score", 0)))

    # Semantic mismatch
    if mismatch.get("mismatch"):
        risk += 0.45
    # Reward strong alignment (retrieval mode)
    if not mismatch.get("mismatch") and dist.get("mean", 0) > 0.74:
        risk -= 0.15


    # Meaning drift (0–1)
    risk += min(0.25, float(meaning_drift.get("score", 0)))

    # System drift
    if drift.get("drift"):
        risk += 0.2

    # Clamp & invert
    risk = max(0.0, min(1.0, risk))
    confidence = round(1.0 - risk, 3)
    return max(0.1, confidence)

def run_single_scan(request: ScanRequest):
    # resolve + validate
    query_vec, doc_vectors = resolve_embeddings(request)
    validate_embeddings(query_vec, doc_vectors)

    # compute similarities
    sims = [cosine_similarity(query_vec, dv) for dv in doc_vectors]

    dist = analyze_distribution(sims)
    redundancy = redundancy_score(doc_vectors)
    mismatch = semantic_mismatch(
    request.query,
    request.retrieved_texts or [],
    query_vec,
    doc_vectors
)

    # drift (if you already have it wired)
    drift = drift_check(dist, request.baseline)

    # decision
    decision = decision_engine(dist, redundancy, mismatch, drift)

    
    # meaning drift: strict only for rewrite mode
    if request.mode == "rewrite" and len(doc_vectors) == 1:
        meaning_drift = meaning_drift_score(query_vec, doc_vectors[0])

        # decision update from meaning drift (rewrite only)
        if meaning_drift["level"] == "high":
            decision["status"] = "error"
            decision["reason_codes"].append("MEANING_DRIFT_HIGH")
        elif meaning_drift["level"] == "medium":
            decision["status"] = "warn"
            decision["reason_codes"].append("MEANING_DRIFT_MEDIUM")

    else:
        meaning_drift = {
            "similarity": None,
            "score": 0.0,
            "level": "low"
        }



    severity = severity_from_decision(decision)
    suggested_actions = suggested_actions_from_reasons(decision["reason_codes"])
    confidence = confidence_score(
    dist=dist,
    redundancy=redundancy,
    mismatch=mismatch,
    meaning_drift=meaning_drift,
    drift=drift
)
    return {
        "num_docs": len(doc_vectors),
        "similarity_scores": sims,
        "distribution": dist,
        "redundancy": redundancy,
        "semantic_mismatch": mismatch,
        "meaning_drift": meaning_drift,
        "decision": decision,
        "severity": severity,
        "suggested_actions": suggested_actions,
        "drift": drift,
        "confidence": confidence,

    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/scan")
def scan(request: ScanRequest):
    return run_single_scan(request)

@app.post("/scan/batch")
def scan_batch(batch: BatchScanRequest):
    results = []
    counts = {"ok": 0, "warn": 0, "error": 0}

    for item in batch.items:
        try:
            res = run_single_scan(item)
            results.append(res)
            counts[res["decision"]["status"]] += 1
        except Exception as e:
            results.append({"error": str(e)})
            counts["error"] += 1

    return {
        "results": results,
        "summary": {
            "total": len(results),
            **counts
        }
    }

