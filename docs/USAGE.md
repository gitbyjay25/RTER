# 📗 RTER — Usage & Integration Guide

RTER is a backend trust & safety service for embedding-based systems (RAG, semantic search, rewrite/humanizer pipelines).
- It runs silently in the background and returns risk signals (confidence, severity, reasons) that your system can act on.

---
## 1. Installation

**Clone the repository:**
```bash
git clone [https://github.com/](https://github.com/)<your-org>/RTER.git
cd RTER
```

---

## 1. Installation

**Clone the repository:**
```bash
git clone [https://github.com/](https://github.com/)<your-org>/RTER.git
cd RTER
```

# Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```
# Install dependencies:
```bash
pip install -r requirements.txt
```
2. Running the Service
Start the server:
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```
- Health check:
```bash
curl http://localhost:8000/health
```

Expected response:
```JSON
{"status":"ok"}
```
3. Core Concept
- RTER does NOT block your application. It only returns:
- decision (ok / warn / error)
- severity
- confidence + confidence_band
- reasons
- suggested_actions
- Your application decides what to do.

4. Retrieval Validation (RAG / Search)
Use this AFTER vector DB retrieval and BEFORE LLM generation.

Example request:
```Bash
curl -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what do cows eat",
    "retrieved_texts": [
      "Cows are herbivores and primarily graze on grass.",
      "A dairy cow diet includes hay and silage."
    ]
  }'
  ```
Example response:
```JSON
{
  "decision": {"status":"ok"},
  "severity": "INFO",
  "confidence": 0.81,
  "confidence_band": "high"
}
```
5. Unsafe Retrieval ExampleRequest:
```Bash
curl -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what do cows eat",
    "retrieved_texts": [
      "Python is a programming language.",
      "Tigers are carnivorous animals."
    ]
  }'
Response:
```JSON{
  "decision": {"status":"error"},
  "severity": "CRITICAL",
  "confidence": 0.18,
  "confidence_band": "unsafe",
  "suggested_actions": ["fallback_to_keyword_search"]
}
```
6. Rewrite / Humanizer ValidationUse AFTER generation but BEFORE final delivery.Request:
```Bash
curl -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{
    "original": "The company allows refunds within seven days.",
    "rewritten": "Customers can request their money back within a week."
  }'
  ```
Response:
```JSON
{
  "decision": {"status":"warn"},
  "severity": "WARN",
  "confidence": 0.74,
  "confidence_band": "medium"
}
```
7. Using Precomputed Embeddings
```Bash
curl -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{
    "query_embedding": [0.12, 0.44, 0.98],
    "doc_embeddings": [
      [0.11, 0.42, 0.97],
      [0.02, 0.91, 0.13]
    ]
  }'
```
8. # Confidence Bands
- Band                         Action
- high                         Proceed
- medium                  Proceed + loglow
- Warn / fallback        unsafeBlock or fallback

9. MetricsPrometheus-compatible metrics:
```Bash
curl http://localhost:8000/metrics
```
## Configuration (Environment Variables)

RTER supports configuration via environment variables.

To customize thresholds:

```bash
cp .env.example .env
```
Edit .env as needed and restart the service.

If no .env is provided, safe defaults are used.