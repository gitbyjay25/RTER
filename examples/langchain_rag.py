import requests

RTER_URL = "http://127.0.0.1:8000/scan"

def fake_retriever(query):
    # simulate retrieved docs from vector DB
    return [
        "Our pricing plans include annual discounts.",
        "Subscription billing details and invoices."
    ]

def call_rter(query, docs, baseline=None):
    payload = {
        "query": query,
        "retrieved_texts": docs
    }
    if baseline:
        payload["baseline"] = baseline

    resp = requests.post(RTER_URL, json=payload)
    resp.raise_for_status()
    return resp.json()

def rag_pipeline(query, baseline=None):
    docs = fake_retriever(query)

    scan = call_rter(query, docs, baseline)
    decision = scan["decision"]["status"]

    if decision == "error":
        print("❌ RTER blocked response:", scan["decision"])
        return "Fallback: keyword search or human review"

    if decision == "warn":
        print("⚠️ RTER warning:", scan["decision"])

    # Normally LLM call would happen here
    return "LLM answer generated from retrieved docs"

if __name__ == "__main__":
    answer = rag_pipeline("refund policy")
    print("Answer:", answer)
