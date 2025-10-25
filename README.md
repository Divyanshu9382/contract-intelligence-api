# Contract Intelligence API

This project implements a FastAPI-based API for ingesting and analyzing contract PDFs using AI, fulfilling the requirements of the AI Developer assignment.

## Features

* **Ingest PDFs**: Upload one or more contract PDFs (`POST /ingest`).
* **Extract Fields**: Extract structured data (parties, dates, clauses, etc.) from contracts (`POST /extract`).
* **Ask Questions (RAG)**: Answer questions based *only* on the content of ingested documents (`POST /ask`).
* **Stream Answers**: Get answers token-by-token for a better user experience (`GET /ask/stream`).
* **Audit Risks**: Identify potentially risky clauses (e.g., unlimited liability, auto-renewal) (`POST /audit`).
* **Monitoring**: Basic health check (`GET /healthz`) and usage metrics (`GET /metrics`).
* **Documentation**: Automatic OpenAPI (Swagger) docs (`GET /docs`).

## Tech Stack

* **Backend**: Python, FastAPI
* **AI/ML**: Google Vertex AI (Gemini 2.5 Flash, Text Embedding 004), LangChain
* **Database**: ChromaDB (Vector Store & Document Store)
* **PDF Parsing**: PyMuPDF (`fitz`)
* **Containerization**: Docker, Docker Compose
* **Testing**: Pytest, HTTPX

## Setup and Running

### Prerequisites

* Docker Desktop installed and running.
* Google Cloud Project with Vertex AI API enabled.
* A Google Cloud Service Account JSON key (`gcp-service-account.json`) with the "Vertex AI User" role.

### Environment Variables

1.  **Create `.env` file**: In the project root, create a file named `.env`.
2.  **Add Project ID**: Add your Google Cloud Project ID to the `.env` file:
    ```
    GCLOUD_PROJECT=your-gcp-project-id-here
    ```
3.  **Add Service Account Key**: Place your downloaded Google Cloud Service Account key file in the project root and name it exactly `gcp-service-account.json`. *(Ensure this file is listed in your `.gitignore`)*.

### Running the Application

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd contract-api
    ```
2.  **Build and Run with Docker Compose**:
    ```bash
    docker compose up --build
    ```
3.  **Access the API**:
    * API Docs (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)
    * Health Check: [http://localhost:8000/healthz](http://localhost:8000/healthz)
    * Metrics: [http://localhost:8000/metrics](http://localhost:8000/metrics)

## API Endpoints & Examples

### `POST /ingest`

Uploads one or more PDF files.

**Example `curl`:**
```bash
curl -X 'POST' \
  'http://localhost:8000/ingest' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@./path/to/your/contract1.pdf;type=application/pdf' \
  -F 'files=@./path/to/your/contract2.pdf;type=application/pdf'
```

**Success Response (200 OK):**
```json
{
  "document_ids": ["uuid-for-contract1", "uuid-for-contract2"]
}
```

### `POST /extract`

Extracts structured fields from an ingested document.

**Example `curl`:**
```bash
# Replace {document_id} with an ID from /ingest
curl -X 'POST' \
  'http://localhost:8000/extract?document_id={document_id}' \
  -H 'accept: application/json'
```

**Success Response (200 OK):**
```json
{
  "parties": ["Party A", "Party B"],
  "effective_date": "2025-10-25",
  "term": "1 year",
  "governing_law": "California",
  "payment_terms": "Net 30 days",
  "termination": "Either party may terminate with 30 days notice",
  "auto_renewal": true,
  "confidentiality": "5 year confidentiality period",
  "indemnity": "Mutual indemnification clause",
  "liability_cap": {
    "number": 100000,
    "currency": "USD"
  },
  "signatories": [
    {
      "name": "John Doe",
      "title": "CEO"
    }
  ]
}
```

### `POST /ask`

Answers a question based on ingested documents.

**Example `curl`:**
```bash
curl -X 'POST' \
  'http://localhost:8000/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What is the governing law?"
}'
```

**Success Response (200 OK):**
```json
{
  "answer": "The agreement is governed by the laws of the State of California.",
  "citations": [
    {
      "document_id": "some-uuid",
      "filename": "contract1.pdf",
      "chunk_number": 5
    }
  ]
}
```

### `POST /audit`

Audits a document for risky clauses.

**Example `curl`:**
```bash
# Replace {document_id} with an ID from /ingest
curl -X 'POST' \
  'http://localhost:8000/audit?document_id={document_id}' \
  -H 'accept: application/json'
```

**Success Response (200 OK):**
```json
{
  "document_id": "{document_id}",
  "findings": [
    {
      "clause": "Confidentiality Clause",
      "evidence": "The receiving party agrees to maintain confidentiality for a period of 5 years",
      "severity": "Low",
      "explanation": "Standard confidentiality period, not excessive"
    },
    {
      "clause": "Auto-Renewal",
      "evidence": "This agreement shall automatically renew unless terminated with 30 days notice",
      "severity": "Medium",
      "explanation": "Auto-renewal with short notice period may be risky"
    }
  ]
}
```

### `GET /ask/stream`

Streams the answer to a question token-by-token.

**Example `curl` (Use Command Prompt, not PowerShell):**
```bash
# Replace spaces in question with %20
curl -N "http://localhost:8000/ask/stream?question=What%20is%20the%20term%20of%20this%20agreement?"
```

**Output (Server-Sent Events):**
```
data: {"token": "The"}
data: {"token": " term"}
data: {"token": " is"}
data: {"token": " one"}
data: {"token": " year"}
data: {"token": "."}
data: {"token": "[DONE]"}
```

## Running Tests

Make sure the API is running via `docker compose up -d`.

Install test dependencies (if not already in your local environment):
```bash
pip install pytest pytest-asyncio httpx
```

Run the tests from the project root directory:
```bash
pytest
```

## Trade-offs and Design Choices

**Framework**: Chose FastAPI for its async capabilities, performance, automatic docs, and Pydantic integration, making it well-suited for I/O-bound tasks like interacting with LLMs and databases.

**Database**: Used ChromaDB as it's a simple, self-contained vector database suitable for local development and demonstration via Docker. A production system might use a managed vector DB (like Vertex AI Matching Engine or Pinecone) for scalability and reliability. The same ChromaDB collection (contracts) is used for storing full text for /extract and /audit for simplicity, though separate storage could be considered.

**LLM**: Leveraged Google Vertex AI (Gemini 2.5 Flash and Text Embedding 004) via LangChain for powerful models accessible via service account authentication, suitable for a "production-ish" setup.

**Chunking**: Employed RecursiveCharacterTextSplitter from LangChain, a standard approach for breaking down text while trying to maintain semantic context. Chunk size (1000) and overlap (100) are common defaults but could be tuned based on contract types and LLM context window limits.

**Error Handling**: Implemented HTTPException for proper FastAPI error responses. /extract uses 404 for missing documents, while other endpoints may return 500 for system errors. This provides clear HTTP status codes for different error conditions.

**Streaming**: Used Server-Sent Events (SSE) via FastAPI's StreamingResponse for /ask/stream as it's simpler to implement than WebSockets for unidirectional streaming from server to client.

**Metrics**: Implemented very basic in-memory counters with thread-safe access. A production system would use dedicated monitoring tools (e.g., Prometheus, Grafana, Cloud Monitoring).

## Limitations

**Scalability**: The current setup runs locally via Docker Compose and isn't designed for high concurrency or large datasets.

**PDF Complexity**: PyMuPDF is generally robust but might struggle with complex layouts, scanned images (OCR not implemented), or password-protected PDFs.

**Extraction/Audit Accuracy**: Performance heavily depends on the LLM's capabilities and the quality of the prompts. Rule-based fallbacks were not explicitly implemented in this version but could be added as improvements.

**Security**: Basic setup. No robust authentication/authorization implemented for API endpoints beyond Google Cloud auth for the LLM. PII redaction for logs is mentioned but not explicitly implemented.

**Testing**: Basic happy-path and some error tests included. More comprehensive testing (edge cases, load testing) would be needed for production.

**Memory Usage**: In-memory metrics and lack of document cleanup could lead to memory issues with large numbers of documents in a long-running instance.