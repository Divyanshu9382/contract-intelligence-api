# Contract Intelligence API

I built this API to analyze contract PDFs using AI. You can upload contracts, ask questions about them, extract key info, and find risky clauses.

## What it does

* **Upload PDFs** - `POST /ingest` - throw your contract PDFs at it
* **Extract data** - `POST /extract` - pulls out parties, dates, terms, etc.
* **Ask questions** - `POST /ask` - RAG-based Q&A that only uses your documents
* **Stream answers** - `GET /ask/stream` - same as above but streams the response
* **Find risks** - `POST /audit` - spots dangerous clauses like unlimited liability
* **Health check** - `GET /healthz` - is it working?
* **Usage stats** - `GET /metrics` - how much you've used it
* **API docs** - `GET /docs` - Swagger UI for testing

## What I used to build it

* **FastAPI** - for the web API (way better than Flask)
* **Google Vertex AI** - Gemini 2.5 Flash for chat, Text Embedding 004 for vectors
* **LangChain** - to glue everything together
* **ChromaDB** - vector database for storing document chunks
* **PyMuPDF** - for extracting text from PDFs
* **Docker** - so you don't have to deal with Python dependencies
* **Pytest** - for testing everything

## How to run it

### What you need first

* Docker Desktop running
* Google Cloud project with Vertex AI enabled
* Service account key with "Vertex AI User" permissions

### Setup

1. **Make a `.env` file** with your GCP project:
   ```
   GCLOUD_PROJECT=your-project-id
   ```

2. **Drop your service account key** in the root folder as `gcp-service-account.json`
   (Don't commit this file!)

3. **Start everything:**
   ```bash
   docker compose up --build
   ```

4. **Check it's working:**
   * API docs: http://localhost:8000/docs
   * Health check: http://localhost:8000/healthz
   * Metrics: http://localhost:8000/metrics

## How to use the API

I'll show you the main endpoints with actual curl examples.

### Upload a contract

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "files=@your-contract.pdf"
```

You get back document IDs:
```json
{
  "document_ids": ["abc-123-def"]
}
```

### Extract key info

```bash
curl -X POST "http://localhost:8000/extract?document_id=abc-123-def"
```

Gets you structured data:
```json
{
  "parties": ["Acme Corp", "Widget Inc"],
  "effective_date": "2024-01-15",
  "term": "2 years",
  "governing_law": "Delaware",
  "auto_renewal": false,
  "liability_cap": {
    "number": 50000,
    "currency": "USD"
  }
}
```

### Ask questions

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What happens if someone breaches this contract?"}'
```

You get answers with sources:
```json
{
  "answer": "The breaching party must pay damages and legal fees...",
  "citations": [
    {
      "document_id": "abc-123-def",
      "filename": "contract.pdf",
      "chunk_number": 3
    }
  ]
}
```

### Find risky stuff

```bash
curl -X POST "http://localhost:8000/audit?document_id=abc-123-def"
```

Spots dangerous clauses:
```json
{
  "document_id": "abc-123-def",
  "findings": [
    {
      "clause": "Unlimited Liability",
      "evidence": "Party shall be liable for all damages without limit",
      "severity": "High",
      "explanation": "No cap on damages - very risky"
    }
  ]
}
```

### Stream responses

```bash
curl -N "http://localhost:8000/ask/stream?question=What%20are%20the%20payment%20terms?"
```

Gets you real-time streaming (good for UIs):
```
data: {"token": "Payment"}
data: {"token": " is"}
data: {"token": " due"}
data: {"token": " within"}
data: {"token": " 30"}
data: {"token": " days"}
data: {"token": "[DONE]"}
```

## Testing

First make sure it's running:
```bash
docker compose up -d
```

Then run the tests:
```bash
pytest
```

Or test the RAG evaluation:
```bash
python eval/evaluate_rag.py
```

## Why I built it this way

**FastAPI** - Great for APIs, auto-generates docs, handles async well

**ChromaDB** - Simple vector DB that just works. For production I'd probably use something managed like Pinecone

**Google Vertex AI** - Reliable, good models, proper enterprise auth

**LangChain** - Makes it easy to chain LLM calls and handle embeddings

**Docker** - No dependency hell, easy to deploy

## Current limitations

- **No auth** - Anyone can use the API
- **Local only** - Runs on your machine, not scalable
- **Basic error handling** - Could be more robust
- **Google quotas** - Easy to hit rate limits
- **PDF parsing** - Struggles with complex layouts or scanned docs
- **Memory usage** - Keeps everything in RAM
- **No PII redaction** - Logs might contain sensitive data

## What I'd add for production

- API authentication (keys, OAuth, etc.)
- Better error handling and retries
- Horizontal scaling
- Proper logging and monitoring
- PII detection and masking
- Rule-based fallbacks for extraction
- Better chunk size optimization
- Database persistence
- Rate limiting
- Health checks and metrics

## Evaluation results

The RAG system currently scores around 20% on my test questions. Not amazing, but it's working - the failures are mostly due to strict keyword matching rather than wrong answers.

Check out `eval/eval_summary.txt` for the latest results.