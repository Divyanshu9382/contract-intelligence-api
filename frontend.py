import streamlit as st
import requests
import os

# Inside Docker, the API service is reachable at http://api:8000
API_URL = os.getenv("BACKEND_URL", "http://api:8000")

st.set_page_config(page_title="Contract Intelligence AI", layout="wide")

st.title("📄 Contract Intelligence AI")
st.markdown("### *Hybrid Search & Proactive Risk Detection*")

# Sidebar: Connection Status & Upload
with st.sidebar:
    st.header("Settings")
    try:
        health = requests.get(f"{API_URL}/healthz", timeout=2).json()
        st.success("✅ API Connected")
    except:
        st.error("❌ API Offline - Check Logs")

    st.divider()
    
    st.header("1. Upload Contract")
    uploaded_files = st.file_uploader(
        "Upload PDFs", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("🚀 Ingest All Documents"):
        if not uploaded_files:
            st.error("Please upload some PDFs first!")
        else:
            for file in uploaded_files:
                with st.spinner(f"Ingesting {file.name}..."):
                    files = {"file": (file.name, file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_URL}/ingest-single", files=files)
                    
                    if response.status_code == 200:
                        st.session_state.doc_id = response.json()["document_id"]
                        # Add to ingested files list
                        if "ingested_files" not in st.session_state:
                            st.session_state.ingested_files = set()
                        st.session_state.ingested_files.add(file.name)
                        st.success(f"✅ {file.name} is now in the database!")
                    else:
                        st.error(f"❌ Failed to ingest {file.name}")
            st.success("All documents processed!")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Knowledge Base")

# Initialize ingested files tracking
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

# Display the list
if st.session_state.ingested_files:
    for doc in st.session_state.ingested_files:
        st.sidebar.caption(f"📄 {doc}")
else:
    st.sidebar.info("No documents in memory yet.")

# Main Dashboard
if "doc_id" in st.session_state:
    tab1, tab2 = st.tabs(["🔍 Smart Query", "🚩 Risk Audit"])

    with tab1:
        st.subheader("🔍 Smart Query")
        
        # Search scope selection
        search_mode = st.radio(
            "Search Scope:",
            ["Current Document Only", "All Uploaded Contracts (Global)"],
            index=0,
            horizontal=True
        )
        
        query = st.text_input("Ask a question about your contract(s):")
        if st.button("Search"):
            if not query:
                st.warning("Please enter a question.")
            else:
                # If user picked Global, we send None as the document_id
                doc_id_to_send = st.session_state.get("doc_id") if search_mode == "Current Document Only" else None
                
                with st.spinner("Analyzing across documents..."):
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"question": query, "document_id": doc_id_to_send}
                    )
                    if response.status_code == 200:
                        res = response.json()
                        st.info(res["answer"])
                        
                        with st.expander("View Source Citations"):
                            for cite in res["citations"]:
                                st.caption(f"Page/Chunk {cite['chunk_number']} from {cite['filename']}")
                    else:
                        st.error(f"Error: {response.text}")

    with tab2:
        if st.button("Run Full Risk Audit"):
            with st.spinner("AI is reviewing legal risks..."):
                audit_res = requests.get(f"{API_URL}/audit/{st.session_state.doc_id}").json()
                st.markdown(audit_res["risks"])
else:
    st.info("👈 Please upload and ingest a contract to begin.")