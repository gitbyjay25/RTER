# RTER
Real-Time Embedding &amp; Retrieval Error Scanner

## 🔌 LangChain / RAG Integration

See `examples/langchain_rag.py` for a drop-in example showing how to:
- Call RTER before LLM generation
- Block or fallback on `error`
- Log on `warn`


## 🚀 Quick Usage

Run the service:

uvicorn api.server:app --reload

