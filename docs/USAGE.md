
---

# 📗 `docs/USAGE.md`  
👉 **Complete HOW-TO guide**

```md
# RTER – Usage & Installation Guide

This document explains how to **install, run, and integrate RTER** into any application.

---

## 📦 Installation Options

### Option 1: Local (Python)

```bash
pip install -r requirements.txt
uvicorn api.server:app --reload


---

# 📗 `docs/USAGE.md`  
👉 **Complete HOW-TO guide**

```md
# RTER – Usage & Installation Guide

This document explains how to **install, run, and integrate RTER** into any application.

---

🚀 Installation & Running
Option 1: Run Locally (Python)
bash
Copy code
pip install -r requirements.txt
uvicorn api.server:app --reload
Access:

Swagger UI: http://127.0.0.1:8000/docs

Health: http://127.0.0.1:8000/health

Metrics: http://127.0.0.1:8000/metrics

Option 2: Docker (Recommended)
bash
Copy code
docker build -t rter .
docker run -p 8000:8000 rter
🔌 API Usage
Endpoint
http
Copy code
POST /scan
Request Example
json
Copy code
{
  "query": "refund policy",
  "retrieved_texts": [
    "Refunds are issued within 7 days for subscriptions.",
    "Our refund policy applies to annual plans."
  ],
  "baseline": {
    "mean": 0.72,
    "std": 0.08
  }
}
Notes
baseline is optional

If not provided, RTER automatically learns a rolling baseline

RTER does not generate embeddings for you — it evaluates behavior

📤 Response Example
json
Copy code
{
  "decision": {
    "status": "error",
    "reason_codes": ["SEMANTIC_MISMATCH"]
  },
  "severity": "CRITICAL",
  "slo": {
    "should_alert": true,
    "should_page": true,
    "error_budget_impact": 1
  }
}
🧭 How Your Application Should React
Decision	Meaning	Recommended Action
ok	Retrieval healthy	Proceed normally
warn	Risk detected	Log, rerank, or fallback
error	Retrieval broken	Block generation / alert

🧠 Example Integration (Pseudo-code)
python
Copy code
scan = call_rter(query, retrieved_docs)

status = scan["decision"]["status"]

if status == "error":
    fallback_to_keyword_search()
elif status == "warn":
    log_warning(scan)
else:
    generate_llm_answer()
⚙️ Configuration (Environment Variables)
bash
Copy code
REDUNDANCY_HIGH=0.9
MISMATCH_SIM=0.6
DRIFT_TOL_MEAN=0.08
DRIFT_TOL_STD=0.05
BASELINE_WINDOW=50
These allow dataset-specific tuning without code changes.

📊 Metrics & Observability
Metrics are exposed at:

bash
Copy code
/metrics
Key metrics:

rter_requests_total

rter_drift_total

rter_scan_latency_seconds

These can be scraped by Prometheus and visualized in Grafana.

🩺 Healthcheck
http
Copy code
GET /health
Response:

json
Copy code
{ "status": "ok" }
Used by:

Load balancers

Kubernetes

Cloud platforms


