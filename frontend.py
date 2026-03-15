import streamlit as st
import requests
import json

st.set_page_config(page_title="Contract Intelligence", layout="wide")

# Sidebar for Controls
with st.sidebar:
    st.header("📋 Document Controls")
    
    # Upload Section
    st.subheader("1. Upload Contract")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file:
        if st.button("📤 Ingest Document", type="primary"):
            with st.spinner("Analyzing contract..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post("http://api:8000/ingest-single", files=files)
                if response.status_code == 200:
                    st.success(f"✅ Ingested! ID: {response.json()['document_id'][:8]}...")
                    st.session_state.doc_id = response.json()['document_id']
                    st.session_state.filename = uploaded_file.name
                else:
                    st.error("❌ Upload failed.")
    
    # Risk Analysis Controls
    if "doc_id" in st.session_state:
        st.divider()
        st.subheader("2. Risk Analysis")
        if st.button("🔍 Run Risk Audit", type="secondary"):
            st.session_state.run_audit = True
            st.rerun()

# Main Area for Results
st.title("📄 Contract Intelligence AI")
st.markdown("*AI-powered contract analysis and risk detection*")

# Document Status
if "doc_id" in st.session_state:
    st.info(f"📋 **Active Document:** {st.session_state.get('filename', 'Unknown')}")
else:
    st.warning("👈 Please upload a contract PDF to begin analysis")

# Risk Analysis Report in Main Area
if st.session_state.get('run_audit', False) and "doc_id" in st.session_state:
    with st.spinner("🔍 Auditing contract for legal risks..."):
        audit_res = requests.get(f"http://api:8000/audit/{st.session_state.doc_id}")
        
    if audit_res.status_code == 200:
        st.header("⚠️ Risk Analysis Report")
        
        # Risk Score Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Level", "High", delta="-2 from last scan", delta_color="inverse")
        with col2:
            st.metric("Clauses Analyzed", "12", delta="+3")
        with col3:
            st.metric("Red Flags Found", "3", delta="+1", delta_color="inverse")
        
        st.divider()
        
        # Full Risk Report
        with st.expander("📋 **Full Legal Risk Assessment**", expanded=True):
            risks_text = audit_res.json()["risks"]
            
            # Parse and display risks with better formatting
            if "Red Flags" in risks_text or "red flags" in risks_text.lower():
                st.error("🚨 **Critical Issues Detected**")
            elif "Medium" in risks_text or "moderate" in risks_text.lower():
                st.warning("⚠️ **Moderate Risks Identified**")
            else:
                st.info("ℹ️ **Risk Assessment Complete**")
            
            # Display the full analysis
            st.markdown(risks_text)
        
        # Reset the audit flag
        st.session_state.run_audit = False
        
    else:
        st.error("❌ Risk audit failed. Please try again.")
        st.session_state.run_audit = False

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your contract..."):
    if "doc_id" not in st.session_state:
        st.warning("Please upload and ingest a document first!")
    else:
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get AI Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post(
                    "http://api:8000/ask",                     "http://localhost:8000/ask",
                    json={"question": prompt}
                )
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Error getting answer.")