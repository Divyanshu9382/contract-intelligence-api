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


class Signatory(BaseModel):
    name: Optional[str] = Field(description="Name of the person signing")
    title: Optional[str] = Field(description="Title of the person signing (e.g., CEO, Director)")


class LiabilityCap(BaseModel):
    number: Optional[float] = Field(description="The numeric value of the liability cap")
    currency: Optional[str] = Field(description="The currency (e.g., USD, EUR, INR)")


class ContractDetails(BaseModel):
    """Extracted details from the contract."""
    parties: List[str] = Field(description="List of all parties involved in the contract")
    effective_date: Optional[str] = Field(description="The date the contract becomes effective")
    term: Optional[str] = Field(description="The length of the contract (e.g., '2 years', 'until 2025')")
    governing_law: Optional[str] = Field(description="The jurisdiction's law that governs the contract")
    payment_terms: Optional[str] = Field(description="Details on payment schedules or terms")
    termination: Optional[str] = Field(description="Conditions or clauses for terminating the contract")
    auto_renewal: Optional[bool] = Field(description="Is there an auto-renewal clause?")
    confidentiality: Optional[str] = Field(description="Summary of the confidentiality clause")
    indemnity: Optional[str] = Field(description="Summary of the indemnity clause")
    liability_cap: Optional[LiabilityCap] = Field(description="Details of the liability cap")
    signatories: List[Signatory] = Field(description="List of signatories (name and title)")


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


from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chroma_client, collection, vector_store, llm, embeddings_model

    print("API: Application startup...")
    print("API: Attempting to connect to ChromaDB...")

    retries = 10
    while retries > 0:
        try:
            chroma_client = chromadb.HttpClient(host="chroma", port=8000)
            chroma_client.heartbeat()
            print("API: Successfully connected to ChromaDB.")
            break
        except Exception as e:
            retries -= 1
            print(f"API: Connection to ChromaDB failed. Retrying... ({retries} retries left)")
            time.sleep(5)

    if not chroma_client:
        print("API: CRITICAL: Could not connect to ChromaDB after all retries.")
        yield
        return

    # Delete old collections to avoid dimension mismatch
    try:
        print("API: Attempting to delete existing 'contract_chunks' collection (if any)...")
        chroma_client.delete_collection(name="contract_chunks")
        print("API: Successfully deleted existing 'contract_chunks' collection.")
    except Exception as e:
        print(f"API: Info - Could not delete 'contract_chunks' collection (may not exist): {e}")

    try:
        print("API: Attempting to delete existing 'contracts' collection (if any)...")
        chroma_client.delete_collection(name="contracts")
        print("API: Successfully deleted existing 'contracts' collection.")
    except Exception as e:
        print(f"API: Info - Could not delete 'contracts' collection (may not exist): {e}")

    # Initialize models
    try:
        llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0)
        print("API: LLM initialized successfully.")
    except Exception as e:
        print(f"API: ERROR initializing LLM: {e}")
        llm = None

    try:
        embeddings_model = VertexAIEmbeddings(model_name="text-embedding-004")  # 768 dimensions
        print("API: Embeddings model initialized successfully.")
    except Exception as e:
        print(f"API: ERROR initializing embeddings model: {e}")
        embeddings_model = None

    # Get the collection for full documents
    try:
        collection = chroma_client.get_or_create_collection(name="contracts")
        print("API: 'contracts' collection initialized.")
    except Exception as e:
        print(f"API: ERROR initializing contracts collection: {e}")
        collection = None

    # Initialize the LangChain Chroma vector store wrapper for RAG
    try:
        vector_store = Chroma(
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


@app.post("/ingest", tags=["Document Ingestion"])
async def ingest_documents(files: List[UploadFile] = File(...)):
    global metrics
    if not collection or not embeddings_model or not vector_store:
        return {"error": "Database not connected. Please check server logs."}

    document_ids = []
    processed_count = 0

    for file in files:
        try:
            doc_id = str(uuid.uuid4())
            file_content = await file.read()

            print(f"\n--- Processing file: {file.filename} ---")
            
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            full_text = ""
            for page in pdf_document:
                full_text += page.get_text()
            pdf_document.close()

            print(f"Extracted text length: {len(full_text)} characters")

            # Store full document
            collection.add(
                documents=[full_text],
                metadatas=[{"filename": file.filename, "doc_id": doc_id}],
                ids=[doc_id]
            )

            # Create chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_text(full_text)
            print(f"Created {len(chunks)} chunks")

            # Prepare chunk data
            chunk_ids = []
            chunk_metadatas = []
            for i, chunk in enumerate(chunks):
                chunk_ids.append(f"{doc_id}_{i}")
                chunk_metadatas.append({
                    "document_id": doc_id,
                    "filename": file.filename,
                    "chunk_number": i
                })

            # Store chunks using LangChain Chroma wrapper
            vector_store.add_texts(
                texts=chunks,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            print(f"Added {len(chunks)} chunks to vector store")
            
            # DEBUG: Test retrieval immediately after ingestion
            test_results = vector_store.similarity_search(
                "National Archives",
                k=3
            )
            print(f"DEBUG: Test query found {len(test_results)} chunks")
            if test_results:
                print(f"First result: {test_results[0].page_content[:100]}...")
            
            document_ids.append(doc_id)
            processed_count += 1
            print(f"Successfully ingested document {doc_id}")

        except Exception as e:
            print(f"ERROR processing file {file.filename}: {e}")
            return {"error": f"Failed to process file {file.filename}: {str(e)}"}
        finally:
            await file.close()

    if processed_count > 0:
        with metrics_lock:
            metrics["documents_ingested"] += processed_count

    return {"document_ids": document_ids}


@app.post("/extract", response_model=ContractDetails, tags=["Document Analysis"])
def extract_contract_details(document_id: str):
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


@app.post("/ask", response_model=AskResponse, tags=["Document Analysis"])
def ask_question(request: AskRequest):
    global metrics
    if not vector_store or not llm or not embeddings_model:
        print(f"DEBUG: System check failed - vector_store: {vector_store is not None}, llm: {llm is not None}, embeddings_model: {embeddings_model is not None}")
        raise HTTPException(status_code=503, detail="System not initialized. Please check server logs.")

    try:
        print(f"\n--- Asking: {request.question} ---")
        
        # Use LangChain similarity search
        results = vector_store.similarity_search_with_score(
            request.question,
            k=5
        )

        print("Retrieved Chunks (Metadatas):")
        if results:
            for doc, score in results:
                print(f"- Doc: {doc.metadata.get('filename')}, Chunk: {doc.metadata.get('chunk_number')}, Score: {score:.3f}")
        else:
            print("- None")

        print("\nRetrieved Chunks (Content):")
        if results:
            for i, (doc, score) in enumerate(results):
                print(f"Chunk {i+1}: {doc.page_content[:150]}...")
        else:
            print("- None")

        if not results:
            print("--> No relevant chunks found.")
            return AskResponse(answer="No relevant information found.", citations=[])

        context = "\n\n".join([doc.page_content for doc, _ in results])

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following context to answer the question. If you don't know the answer, say so."),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])

        chain = prompt | llm
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


@app.post("/audit", response_model=AuditResponse, tags=["Document Analysis"])
def audit_document(document_id: str):
    """Audits a document for risky clauses."""
    if not llm:
        return AuditResponse(
            document_id=document_id,
            findings=[RiskFinding(
                clause="System Error",
                evidence="LLM not initialized",
                severity=Severity.HIGH,
                explanation="System configuration error - LLM not available"
            )]
        )

    try:
        doc = collection.get(ids=[document_id])
        if not doc or not doc.get('documents'):
            return AuditResponse(
                document_id=document_id,
                findings=[RiskFinding(
                    clause="Document Error",
                    evidence="Document not found",
                    severity=Severity.HIGH,
                    explanation="The specified document could not be retrieved"
                )]
            )
        full_text = doc['documents'][0]
    except Exception as e:
        return AuditResponse(
            document_id=document_id,
            findings=[RiskFinding(
                clause="Retrieval Error",
                evidence=str(e),
                severity=Severity.HIGH,
                explanation="Failed to retrieve document from database"
            )]
        )

    parser = PydanticOutputParser(pydantic_object=AuditFindings)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an expert legal reviewer. Audit the contract for risky clauses:
         - HIGH severity: Unlimited Liability clauses
         - MEDIUM severity: Auto-Renewal clauses (especially <60 days notice), broad Indemnity clauses
         - LOW severity: Confidentiality clauses >5 years or indefinite
         
         For each finding, provide clause type, exact evidence, severity, and explanation.
         {format_instructions}"""),
        ("human", "Contract Text: \n\n{contract_text}")
    ])

    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "contract_text": full_text,
            "format_instructions": parser.get_format_instructions()
        })
        return AuditResponse(
            document_id=document_id,
            findings=result.findings
        )
    except Exception as e:
        return AuditResponse(
            document_id=document_id,
            findings=[RiskFinding(
                clause="Analysis Error",
                evidence=str(e),
                severity=Severity.HIGH,
                explanation="Failed to analyze document for risks"
            )]
        )


async def stream_rag_response(context: str, question: str) -> AsyncGenerator[str, None]:
    """Async generator to stream the RAG response as Server-Sent Events (SSE)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following context to answer the question. If you don't know the answer, say so."),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    
    chain = prompt | llm
    
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
        question_embedding = embeddings_model.embed_query(question)
        
        results = vector_store.query(
            query_embeddings=[question_embedding],
            n_results=5
        )
        
        if not results['documents'] or not results['documents'][0]:
            async def empty_stream():
                yield f"data: {json.dumps({'token': 'No relevant information found.'})}\n\n"
                yield f"data: {json.dumps({'token': '[DONE]'})}\n\n"
                
            return StreamingResponse(empty_stream(), media_type="text/event-stream")
        
        context = "\n\n".join(results['documents'][0])
        
        with metrics_lock:
            metrics["questions_streamed"] += 1
        
        return StreamingResponse(
            stream_rag_response(context, question), 
            media_type="text/event-stream"
        )
    except Exception as e:
        async def error_stream():
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")