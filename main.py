import streamlit as st
from utils.file_processor import FileProcessor
from agents.retriever import Retriever
from agents.generator import Generator
from agents.verifier import Verifier
from dotenv import load_dotenv
import os

def initialize_components():
    """Initialize application components once per session"""
    load_dotenv()
    
    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY in .env file")
        st.stop()
    
    return {
        "file_processor": FileProcessor(),
        "retriever": Retriever(),
        "generator": Generator(),
        "verifier": Verifier()
    }

# Initialize components in session state
if 'components' not in st.session_state:
    st.session_state.components = initialize_components()

# Main application UI
st.title("Multi-Agent RAG System")
components = st.session_state.components

# Document processing
uploaded_file = st.file_uploader("Upload document (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])
processed_data = None

if uploaded_file:
    with st.spinner("Analyzing document..."):
        processed_data = components["file_processor"].process_uploaded_file(uploaded_file)
        components["retriever"].index_documents(processed_data["text_sections"])
        st.success(f"Processed {len(processed_data['text_sections'])} text sections and {len(processed_data['image_metadata'])} images")

# Question handling
question = st.text_input("Ask about the document")
if question and processed_data:
    with st.spinner("Generating answer..."):
        context = components["retriever"].retrieve_relevant_text(question)
        response = components["generator"].generate_response(question, context, processed_data["image_metadata"])
        verification = components["verifier"].check_hallucinations(context, response)
        
        st.subheader("Answer")
        if not verification["is_consistent"]:
            st.error("⚠️ Potential inaccuracies detected")
        st.write(response)
        
        with st.expander("Verification details"):
            st.json(verification)