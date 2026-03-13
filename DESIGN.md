# Contract API Design Doc

Here's how I built this thing and why I made the choices I did.

## How It All Fits Together

```
You (curl/browser) → FastAPI App → Google AI
                         ↓
                    ChromaDB
```

Basically:
- You send requests to my FastAPI app
- App talks to Google's AI for the smart stuff
- ChromaDB stores all the document chunks and vectors
- Everything runs in Docker containers so it's easy to set up

**What each piece does:**

**FastAPI App** - This is the main thing. It:
- Takes your PDF uploads and questions
- Breaks PDFs into chunks using PyMuPDF
- Talks to Google AI for embeddings and chat
- Stores everything in ChromaDB
- Gives you back answers with citations

**Google Vertex AI** - The brain of the operation:
- Gemini 2.5 Flash for answering questions and extracting data
- Text Embedding 004 for turning text into vectors (768 dimensions)
- Need a service account key to use it

**ChromaDB** - The memory:
- `contracts` collection: full document text for /extract and /audit
- `contract_chunks` collection: small pieces with embeddings for RAG search
- Each chunk knows which document it came from

## How I Store Everything

I keep two collections in ChromaDB:

**contracts** - The whole documents
- ID: random UUID I generate
- Content: entire PDF text after extraction
- Used for: /extract and /audit endpoints that need the full context
- Metadata: just filename and doc_id

**contract_chunks** - Small pieces for search
- ID: `{document_id}_{chunk_number}` like "abc123_0", "abc123_1"
- Content: ~1000 character chunks of text
- Embeddings: 768-dimension vectors from Google's embedding model
- Used for: /ask endpoint to find relevant pieces
- Metadata: which doc it came from, filename, chunk number (for citations)

No fancy database needed - ChromaDB handles everything including the vector search.

## Why I Chunk Text This Way

I use LangChain's `RecursiveCharacterTextSplitter` with:
- **chunk_size=1000** - About a paragraph or two
- **chunk_overlap=100** - So sentences don't get cut in half

Why these numbers?
- 1000 chars is big enough to have context but small enough to be specific
- 100 char overlap means if something important spans chunks, we don't lose it
- The splitter tries to break on paragraphs first, then sentences, so it's not random

Could probably tune these better for contracts specifically, but these defaults work pretty well.

## What Happens When Things Break

**Google AI fails** - Right now I just return 500 errors with HTTPException. In real life I'd add retries and maybe cache some responses.

**ChromaDB goes down** - The app tries 10 times to connect on startup, then gives up. If it fails during a request, you get an error.

**Rule engine fallback** - I didn't build this but could add regex patterns to catch basic stuff like:
- Dates with regex patterns
- Dollar amounts with `$\d+` matching
- Common risky phrases like "unlimited liability"

Would be faster and cheaper than hitting the LLM every time.

## Security Stuff

**Google Cloud auth** - I use a service account JSON key that gets mounted into the Docker container. Keep this file secret and don't commit it.

**API protection** - There's none right now. Anyone can hit the endpoints. In production I'd add API keys or OAuth.

**Input validation** - FastAPI + Pydantic handles most of this automatically. File uploads use FastAPI's built-in stuff.

**Secrets** - GCP project ID is in .env file, service account key is a separate JSON file. Both are gitignored.

**PII handling** - I don't redact anything from logs yet. Contracts have sensitive info so this would be important for production.

**Data storage** - Everything sits in Docker volumes locally. For production you'd want encryption and proper access controls.

## Current Limitations

- No authentication on the API
- No PII redaction in logs  
- No rule-based fallbacks
- Basic error handling
- Runs locally only
- Hit Google's quota limits pretty easily
- Evaluation keywords are pretty strict

## What I'd Add Next

- API key authentication
- Better error handling with retries
- Rule-based extraction fallbacks
- PII detection and masking
- Rate limiting
- Better chunk size tuning for contracts
- More flexible evaluation criteria