from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, AsyncGenerator
import uuid
import fitz  # PyMuPDF
import chromadb
import time
from contextlib import asynccontextmanager
import os
from enum import Enum
from fastapi.responses import StreamingResponse
import json
import threading
import re

# LangChain & AI Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document


# --- Pydantic Models ---

class Signatory(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None


class LiabilityCap(BaseModel):
    number: Optional[float] = None
    currency: Optional[str] = None


class ContractDetails(BaseModel):
    parties: List[str] = []
    effective_date: Optional[str] = None
    term: Optional[str] = None
    governing_law: Optional[str] = None
    payment_terms: Optional[str] = None
    termination: Optional[str] = None
    auto_renewal: Optional[bool] = None
    confidentiality: Optional[str] = None
    indemnity: Optional[str] = None
    liability_cap: Optional[LiabilityCap] = None
    signatories: List[Signatory] = []


class AskRequest(BaseModel):
    question: str
    document_id: Optional[str] = None  # None = Global Search


class Citation(BaseModel):
    document_id: str
    filename: str
    chunk_number: int


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]


class Severity(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class RiskFinding(BaseModel):
    clause: str
    evidence: str
    severity: Severity
    explanation: str


class AuditResponse(BaseModel):
    document_id: str
    risks: str  # Markdown text for simple display


# --- Global State ---
chroma_client = None
collection = None
vector_store = None
llm = None
embeddings_model = None


# --- Lifecycle Management ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chroma_client, collection, vector_store, llm, embeddings_model
    
    print("API: Starting up...")
    print("API: Connecting to ChromaDB...")
    
    retries = 10
    connected = False
    
    while retries > 0:
        try:
            # Standard connection now that versions are aligned
            chroma_client = chromadb.HttpClient(host="chroma", port=8000)
            chroma_client.heartbeat()
            print("API: Connected to ChromaDB!")
            connected = True
            break
        except Exception as e:
            retries -= 1
            print(f"API: Waiting for ChromaDB... ({retries} left). Error: {e}")
            time.sleep(5)
    
    if not connected:
        print("API: CRITICAL: Could not connect to ChromaDB.")
        yield
        return

    # Initialize models (Ensuring GOOGLE_API_KEY is in your .env)
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        collection = chroma_client.get_or_create_collection(name="contracts")
        vector_store = Chroma(
            client=chroma_client,
            collection_name="contract_chunks",
            embedding_function=embeddings_model
        )
        print("API: RAG pipeline initialized successfully.")
    except Exception as e:
        print(f"API: Model Setup Error: {e}")
    
    yield
    print("API: Application shutting down.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---

@app.post("/ingest-single")
async def ingest_single(file: UploadFile = File(...)):
    if not vector_store:
        raise HTTPException(status_code=503, detail="System not ready")

    # Use filename as unique ID to prevent overwriting
    doc_id = file.filename
    content = await file.read()

    pdf = fitz.open(stream=content, filetype="pdf")
    text = "".join(page.get_text() for page in pdf)
    pdf.close()

    # Save full text
    collection.add(documents=[text], metadatas=[{"filename": file.filename}], ids=[doc_id])

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    metadatas = [{"document_id": doc_id, "filename": file.filename, "chunk_number": i} for i in range(len(chunks))]
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

    vector_store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)

    return {"document_id": doc_id, "filename": file.filename}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    # 1. Determine Search Scope
    search_filter = None
    if request.document_id:
        # Single document search
        search_filter = {"document_id": request.document_id}
        print(f"API: Filtering by Document: {request.document_id}")
    else:
        # Cross-document search
        print("API: Performing Global Cross-Document Search")

    try:
        # 2. Query Vector Store with the dynamic filter
        docs = vector_store.similarity_search(
            request.question, 
            k=5, 
            filter=search_filter
        )
        
        # 3. Context Construction (Includes filename so AI knows which is which)
        context = "\n\n".join([
            f"[Source: {d.metadata.get('filename', 'Unknown')}]\n{d.page_content}" 
            for d in docs
        ])
        
        # 4. Generate Answer
        prompt = f"Use the following contract snippets to answer. Mention the filename if comparing documents.\n\nContext:\n{context}\n\nQuestion: {request.question}"
        response = llm.invoke(prompt)
        
        # 5. Create citations from docs
        citations = [Citation(
            document_id=d.metadata.get("document_id", "unknown"),
            filename=d.metadata.get("filename", "Unknown"),
            chunk_number=d.metadata.get("chunk_number", 0)
        ) for d in docs]
        
        return AskResponse(answer=response.content, citations=citations)
        
    except Exception as e:
        print(f"API ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/{document_id}")
def audit_contract(document_id: str):
    doc_data = collection.get(ids=[document_id])
    if not doc_data or not doc_data.get('documents'):
        raise HTTPException(status_code=404, detail="Document not found")

    full_text = doc_data['documents'][0]

    audit_prompt = f"""
    Analyze this contract for 3 key risks:
    1. Liability/Indemnity 2. Termination 3. Auto-renewal.
    Provide findings in clear Markdown.

    Text: {full_text[:10000]}
    """

    response = llm.invoke(audit_prompt)
    return {"risks": response.content}


@app.get("/debug-db")
async def debug_db():
    # This will return a list of all unique filenames currently in the database
    results = collection.get()
    filenames = set([m.get("filename") for m in results["metadatas"]])
    return {"total_chunks": len(results["ids"]), "unique_files": list(filenames)}


@app.get("/healthz")
def health(): return {"status": "ok"}