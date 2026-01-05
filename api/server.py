from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import os
from enum import Enum
from collections import deque

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
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
    query: str
    retrieved_texts: List[str]
    baseline: dict | None = None

def severity_from_decision(decision):
    codes = set(decision.get("reason_codes", []))
    if "SEMANTIC_MISMATCH" in codes:
        return Severity.CRITICAL
    if any(c in codes for c in ["HIGH_REDUNDANCY", "COLLAPSED_DISTRIBUTION", "DRIFT_MEAN_SHIFT", "DRIFT_STD_SHIFT"]):
        return Severity.WARN
    return Severity.INFO


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
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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

    # decide FIRST
    decision = decision_engine(dist, redundancy, mismatch, drift=None)

    # baseline from client OR auto-learned
    auto_baseline = update_and_get_baseline(dist, decision)
    baseline = request.baseline or auto_baseline

    # now check drift
    drift = drift_check(dist, baseline)

    # update decision WITH drift
    decision = decision_engine(dist, redundancy, mismatch, drift)

    severity = severity_from_decision(decision)

    slo = {
        "should_alert": severity in [Severity.WARN, Severity.CRITICAL],
        "should_page": severity == Severity.CRITICAL,
        "error_budget_impact": 1 if severity != Severity.INFO else 0
    }
    start = time.time()

    LATENCY.observe(time.time() - start)
    REQ_COUNTER.labels(
        status=decision["status"],
        severity=severity.value
    ).inc()

    if drift.get("drift"):
        DRIFT_COUNTER.inc()

    return {
        "status": "ok",
        "num_docs": len(request.retrieved_texts),
        "similarity_scores": similarities,
        "distribution": dist,
        "redundancy": redundancy,
        "semantic_mismatch": mismatch,
        "decision": decision,
        "drift": drift,
        "severity": severity,
        "slo": slo





    }
