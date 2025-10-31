from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, AsyncGenerator
import uuid
import fitz
import chromadb
import time
from contextlib import asynccontextmanager
import os
from enum import Enum
from fastapi.responses import StreamingResponse
import json
import threading
import re  # <-- IMPORTED FOR FALLBACK


# --- Pydantic Models ---
# (Your models remain unchanged)
class Signatory(BaseModel):
    name: Optional[str] = Field(description="Name of the person signing")
    title: Optional[str] = Field(description="Title of the person signing (e.g., CEO, Director)")


class LiabilityCap(BaseModel):
    number: Optional[float] = Field(description="The numeric value of the liability cap")
    currency: Optional[str] = Field(description="The currency (e.g., USD, EUR, INR)")


class ContractDetails(BaseModel):
    """all the stuff we want to extract from contracts"""
    parties: List[str] = Field(description="who's involved in this contract")
    effective_date: Optional[str] = Field(description="when does this thing start")
    term: Optional[str] = Field(description="how long does it last")
    governing_law: Optional[str] = Field(description="which state/country laws apply")
    payment_terms: Optional[str] = Field(description="how and when to pay")
    termination: Optional[str] = Field(description="how to end the contract")
    auto_renewal: Optional[bool] = Field(description="does it auto-renew")
    confidentiality: Optional[str] = Field(description="what's the NDA stuff")
    indemnity: Optional[str] = Field(description="who covers damages")
    liability_cap: Optional[LiabilityCap] = Field(description="max liability amount")
    signatories: List[Signatory] = Field(description="who signed it")


class AskRequest(BaseModel):
    question: str


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
    clause: str = Field(description="The type of risky clause found (e.g., 'Unlimited Liability', 'Auto-Renewal')")
    evidence: str = Field(description="The exact text snippet from the contract that constitutes the risk.")
    severity: Severity = Field(description="The assessed severity of the risk (Low, Medium, High).")
    explanation: str = Field(description="A brief explanation of why this clause is a risk.")


class AuditFindings(BaseModel):
    findings: List[RiskFinding] = Field(description="List of risk findings from the audit")


class AuditResponse(BaseModel):
    document_id: str
    findings: List[RiskFinding]


# --- LLM and DB Imports ---
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # This is the LangChain wrapper you imported

# --- Global Variables ---
# (Unchanged)
chroma_client = None
collection = None
vector_store = None
llm = None
embeddings_model = None

metrics = {
    "documents_ingested": 0,
    "questions_asked": 0,
    "questions_streamed": 0
}
metrics_lock = threading.Lock()


# --- NEW: Rule-Based Fallback Function ---
def extract_fields_rules(full_text: str) -> dict:
    """Very basic rule-based extraction for limited fields."""
    extracted = {
        "parties": [], "effective_date": None, "term": None, "governing_law": None,
        "payment_terms": None, "termination": None, "auto_renewal": None,
        "confidentiality": None, "indemnity": None, "liability_cap": None,
        "signatories": []
    }

    # Rule 1: Find Disclosing Party
    match_disclosing = re.search(r"between\s+(.*?),\s*\(the\s*\"Disclosing Party\"\)", full_text, re.IGNORECASE)
    if match_disclosing:
        party_name = match_disclosing.group(1).strip()
        if party_name not in extracted["parties"]:
            extracted["parties"].append(party_name)
        print(f"Rule-based: Found Disclosing Party: {party_name}")

    # Rule 2: Find Receiving Party
    match_receiving = re.search(r"and\s+(.*?),\s*located at.*\(the\s*\"Receiving Party\"\)", full_text,
                                re.IGNORECASE | re.DOTALL)
    if match_receiving:
        party_name = match_receiving.group(1).strip()
        if party_name not in extracted["parties"]:
            extracted["parties"].append(party_name)
        print(f"Rule-based: Found Receiving Party: {party_name}")

    # You could add more regex rules here...

    return extracted


# --- Lifespan Event ---
# (Unchanged)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global chroma_client, collection, vector_store, llm, embeddings_model
    # ... (Your startup logic remains the same) ...
    print("API: Starting up...")
    print("API: Trying to connect to ChromaDB...")
    retries = 10
    while retries > 0:
        try:
            chroma_client = chromadb.HttpClient(host="chroma", port=8000)
            chroma_client.heartbeat()
            print("API: Connected to ChromaDB!")
            break
        except Exception as e:
            retries -= 1
            print(f"API: ChromaDB not ready yet, trying again... ({retries} left)")
            time.sleep(5)
    if not chroma_client:
        print("API: CRITICAL: Could not connect to ChromaDB after all retries.")
        yield
        return
    try:
        print("API: Deleting old contract_chunks collection...")
        chroma_client.delete_collection(name="contract_chunks")
        print("API: Deleted old contract_chunks")
    except Exception as e:
        print(f"API: No old contract_chunks to delete: {e}")
    try:
        print("API: Deleting old contracts collection...")
        chroma_client.delete_collection(name="contracts")
        print("API: Deleted old contracts")
    except Exception as e:
        print(f"API: No old contracts to delete: {e}")
    try:
        llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0)  # Using your selected model
        print("API: Got the LLM working")
    except Exception as e:
        print(f"API: LLM setup failed: {e}")
        llm = None
    try:
        embeddings_model = VertexAIEmbeddings(model_name="text-embedding-004")
        print("API: Embeddings model ready")
    except Exception as e:
        print(f"API: Embeddings setup failed: {e}")
        embeddings_model = None
    try:
        collection = chroma_client.get_or_create_collection(name="contracts")
        print("API: 'contracts' collection initialized.")
    except Exception as e:
        print(f"API: ERROR initializing contracts collection: {e}")
        collection = None
    try:
        vector_store = Chroma(  # This correctly uses the LangChain wrapper
            client=chroma_client,
            collection_name="contract_chunks",
            embedding_function=embeddings_model
        )
        print("API: LangChain 'vector_store' (for contract_chunks) initialized.")
    except Exception as e:
        print(f"API: ERROR initializing vector store: {e}")
        vector_store = None
    print("API: RAG pipeline initialized successfully.")
    yield
    print("API: Application shutting down.")


app = FastAPI(
    title="Contract Intelligence API",
    description="API for ingesting and analyzing contract PDFs.",
    version="0.1.0",
    lifespan=lifespan
)


# --- General Endpoints ---
# (Unchanged)
@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the Contract Intelligence API!"}


@app.get("/healthz", tags=["General"])
def health_check():
    return {"status": "ok"}


@app.get("/metrics", tags=["General"])
def get_metrics():
    """Returns basic usage metrics."""
    with metrics_lock:
        return metrics.copy()


# --- MODIFIED /ingest Endpoint ---
@app.post("/ingest", tags=["Document Ingestion"])
async def ingest_documents(files: List[UploadFile] = File(...)):
    global metrics
    if not collection or not embeddings_model or not vector_store:
        raise HTTPException(status_code=503, detail="System not initialized. Please check server logs.")

    document_ids = []
    processed_count = 0

    for file in files:
        try:
            doc_id = str(uuid.uuid4())
            file_content = await file.read()
            print(f"\n--- Processing file: {file.filename} ---")

            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            full_text = "".join(page.get_text() for page in pdf_document)
            pdf_document.close()
            print(f"Extracted text length: {len(full_text)} characters")

            collection.add(
                documents=[full_text],
                metadatas=[{"filename": file.filename, "doc_id": doc_id}],
                ids=[doc_id]
            )

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_text(full_text)
            print(f"Created {len(chunks)} chunks")

            if not chunks:
                print("No chunks generated, skipping vector store add.")
                document_ids.append(doc_id)
                processed_count += 1
                continue

            chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            chunk_metadatas = [{
                "document_id": doc_id,
                "filename": file.filename,
                "chunk_number": i
            } for i, chunk in enumerate(chunks)]

            # *** CORRECTED METHOD ***
            # Use .add_texts() which is the LangChain Chroma wrapper method
            # This handles the embedding step for you.
            print(f"Adding {len(chunks)} chunks to vector store...")
            vector_store.add_texts(
                texts=chunks,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            print(f"Added {len(chunks)} chunks to vector store")

            # DEBUG: Test retrieval immediately
            test_results = vector_store.similarity_search("National Archives", k=1)  # Use sync search
            print(f"DEBUG: Test query found {len(test_results)} chunks")
            if test_results:
                print(f"First result: {test_results[0].page_content[:100]}...")

            document_ids.append(doc_id)
            processed_count += 1
            print(f"Successfully ingested document {doc_id}")

        except Exception as e:
            print(f"ERROR processing file {file.filename}: {e}")
            # Use HTTPException to return a proper error
            raise HTTPException(status_code=500, detail=f"Failed to process file {file.filename}: {str(e)}")
        finally:
            await file.close()

    if processed_count > 0:
        with metrics_lock:
            metrics["documents_ingested"] += processed_count

    return {"document_ids": document_ids}


# --- MODIFIED /extract Endpoint ---
@app.post("/extract", response_model=ContractDetails, tags=["Document Analysis"])
def extract_contract_details(document_id: str, use_fallback: bool = False):  # <-- ADDED use_fallback
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized.")

    try:
        doc_data = collection.get(ids=[document_id])
        if not doc_data or not doc_data.get('documents') or len(doc_data['documents']) == 0:
            raise HTTPException(status_code=404, detail="Document not found.")
        full_text = doc_data['documents'][0]
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR retrieving document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")

    # --- NEW: Fallback Logic ---
    if use_fallback:
        print(f"Using rule-based fallback extraction for {document_id}")
        try:
            rule_based_result = extract_fields_rules(full_text)
            # We return this directly. FastAPI will validate it against ContractDetails
            return rule_based_result
        except Exception as rule_err:
            print(f"ERROR during rule-based fallback for {document_id}: {rule_err}")
            raise HTTPException(status_code=500, detail=f"Rule-based extraction failed: {str(rule_err)}")

    # --- LLM Logic (if use_fallback is False) ---
    print(f"Using LLM extraction for {document_id}")
    parser = PydanticOutputParser(pydantic_object=ContractDetails)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert legal assistant. Extract the requested fields from the following contract text.\n{format_instructions}"),
        ("human", "Contract Text: \n\n{contract_text}")
    ])
    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "contract_text": full_text,
            "format_instructions": parser.get_format_instructions()
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed during extraction: {str(e)}")


# --- MODIFIED /ask Endpoint ---
@app.post("/ask", response_model=AskResponse, tags=["Document Analysis"])
def ask_question(request: AskRequest):  # This is sync, so we use sync methods
    global metrics
    if not vector_store or not llm or not embeddings_model:
        print(
            f"DEBUG: System check failed - vector_store: {vector_store is not None}, llm: {llm is not None}, embeddings_model: {embeddings_model is not None}")
        raise HTTPException(status_code=503, detail="System not initialized. Please check server logs.")

    try:
        print(f"\n--- Asking: {request.question} ---")

        # *** CORRECTED METHOD ***
        # Use sync .similarity_search_with_score since this is a sync (def) function
        results = vector_store.similarity_search_with_score(
            request.question,
            k=5
        )

        print("Retrieved Chunks (Metadatas):")
        if results:
            for doc, score in results:
                print(
                    f"- Doc: {doc.metadata.get('filename')}, Chunk: {doc.metadata.get('chunk_number')}, Score: {score:.3f}")
        else:
            print("- None")
        # ... (rest of debug prints) ...

        if not results:
            print("--> No relevant chunks found.")
            return AskResponse(answer="No relevant information found.", citations=[])

        context = "\n\n".join([doc.page_content for doc, _ in results])

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks..."),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])

        chain = prompt | llm
        # Use sync .invoke()
        response = chain.invoke({
            "context": context,
            "question": request.question
        })

        citations = []
        for doc, _ in results:
            citations.append(Citation(
                document_id=doc.metadata.get("document_id"),
                filename=doc.metadata.get("filename"),
                chunk_number=doc.metadata.get("chunk_number")
            ))

        with metrics_lock:
            metrics["questions_asked"] += 1

        print(f"LLM Answer: {response.content[:100]}...")
        print("-" * 20)

        return AskResponse(
            answer=response.content,
            citations=citations
        )
    except Exception as e:
        print(f"ERROR during /ask: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")


# --- /audit Endpoint ---
# (Unchanged)
@app.post("/audit", response_model=AuditResponse, tags=["Document Analysis"])
def audit_document(document_id: str):
    # ... (Your sync audit code remains the same) ...
    if not llm:
        return AuditResponse(document_id=document_id, findings=[
            RiskFinding(clause="System Error", evidence="LLM not initialized", severity=Severity.HIGH,
                        explanation="System configuration error - LLM not available")])
    try:
        doc = collection.get(ids=[document_id])
        if not doc or not doc.get('documents'):
            return AuditResponse(document_id=document_id, findings=[
                RiskFinding(clause="Document Error", evidence="Document not found", severity=Severity.HIGH,
                            explanation="The specified document could not be retrieved")])
        full_text = doc['documents'][0]
    except Exception as e:
        return AuditResponse(document_id=document_id, findings=[
            RiskFinding(clause="Retrieval Error", evidence=str(e), severity=Severity.HIGH,
                        explanation="Failed to retrieve document from database")])
    parser = PydanticOutputParser(pydantic_object=AuditFindings)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert legal reviewer...{format_instructions}"""),
        ("human", "Contract Text: \n\n{contract_text}")
    ])
    chain = prompt | llm | parser
    try:
        result = chain.invoke({"contract_text": full_text, "format_instructions": parser.get_format_instructions()})
        return AuditResponse(document_id=document_id, findings=result.findings)
    except Exception as e:
        return AuditResponse(document_id=document_id, findings=[
            RiskFinding(clause="Analysis Error", evidence=str(e), severity=Severity.HIGH,
                        explanation="Failed to analyze document for risks")])


# --- MODIFIED /ask/stream Endpoint ---
async def stream_rag_response(context: str, question: str) -> AsyncGenerator[str, None]:
    """Async generator to stream the RAG response as Server-Sent Events (SSE)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks..."),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    chain = prompt | llm

    # Use async .astream()
    async for chunk in chain.astream({
        "context": context,
        "question": question
    }):
        if chunk.content:
            yield f"data: {json.dumps({'token': chunk.content})}\n\n"

    yield f"data: {json.dumps({'token': '[DONE]'})}\n\n"


@app.get("/ask/stream", tags=["Document Analysis"])
async def ask_question_stream(question: str):
    """Answers a question using RAG, streaming the response as Server-Sent Events (SSE)."""
    global metrics
    if not vector_store or not llm or not embeddings_model:
        async def error_stream():
            yield f"data: {json.dumps({'error': 'System not initialized'})}\n\n"

        return StreamingResponse(error_stream(), media_type="text/event-stream")

    try:
        # *** CORRECTED METHOD ***
        # Use async .asimilarity_search since this is an async (async def) function
        results_docs = await vector_store.asimilarity_search(
            question,
            k=5
        )

        if not results_docs:
            async def empty_stream():
                yield f"data: {json.dumps({'token': 'No relevant information found.'})}\n\n"
                yield f"data: {json.dumps({'token': '[DONE]'})}\n\n"

            return StreamingResponse(empty_stream(), media_type="text/event-stream")

        context = "\n\n".join([doc.page_content for doc in results_docs])

        with metrics_lock:
            metrics["questions_streamed"] += 1

        return StreamingResponse(
            stream_rag_response(context, question),
            media_type="text/event-stream"
        )
    except Exception as e:
        async def error_stream(error_message: str):  # Pass message
            yield f"data: {json.dumps({'error': error_message})}\n\n"
            yield f"data: {json.dumps({'token': '[DONE]'})}\n\n"

        return StreamingResponse(error_stream(str(e)), media_type="text/event-stream")