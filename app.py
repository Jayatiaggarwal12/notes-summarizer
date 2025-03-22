import streamlit as st
st.set_page_config(
    page_title="DocAnalyzer Pro",
    page_icon="📘",
    layout="wide"
)

import os
import nltk
import numpy as np
import faiss
import fitz
import re
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64
import docx
import tiktoken
from typing import List, Tuple, Dict

# Initialize environment
nltk.download(['punkt'])
load_dotenv()

# Configuration
class Config:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "llama3-70b-8192"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K = 3
    MAX_TOKENS = 4000
    TEMPERATURE = 0.3

# Session state management
def init_session_state():
    return {
        'chat_history': [],
        'document_processed': False,
        'text_chunks': [],
        'faiss_index': None,
        'full_text': "",
        'summaries': {},
        'key_points': [],
        'suggested_questions': [],
        'pdf_buffer': None,
        'processing_error': None
    }

if 'app_state' not in st.session_state:
    st.session_state.app_state = init_session_state()

# Model loading
@st.cache_resource
def load_models():
    try:
        embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        llm = ChatGroq(
            model_name=Config.LLM_MODEL,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=Config.TEMPERATURE
        )
        return embedding_model, llm
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

embedding_model, llm = load_models()

# Text processing utilities
def chunk_text(text: str) -> List[str]:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), Config.CHUNK_SIZE):
        chunk = tokens[i:i + Config.CHUNK_SIZE]
        chunks.append(tokenizer.decode(chunk))
    
    return chunks

def safe_extract_text(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            return "\n".join([page.get_text() for page in doc])
        elif uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        return ""
    except Exception as e:
        st.session_state.app_state['processing_error'] = f"Text extraction error: {str(e)}"
        return ""

# AI processing functions
def map_reduce_summary(chunks: List[str]) -> str:
    try:
        map_template = """Write a concise summary of this text chunk:
        {text}
        CONCISE SUMMARY:"""
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=llm, prompt=map_prompt)

        reduce_template = """Combine these summaries into a final comprehensive summary:
        {summaries}
        FINAL SUMMARY:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        return reduce_chain.run([map_chain.run(chunk) for chunk in chunks])
    except Exception as e:
        return f"Summary generation failed: {str(e)}"

def generate_key_points(text: str) -> List[str]:
    try:
        prompt = PromptTemplate.from_template("""
        Extract 5-7 key points from this text. Format as bullet points:
        {text}
        KEY POINTS:
        """)
        result = LLMChain(llm=llm, prompt=prompt).run({"text": text})
        return [p.strip() for p in result.split('\n') if p.strip().startswith('-')]
    except Exception as e:
        return [f"Key points error: {str(e)}"]

# UI Components
def sidebar_content():
    with st.sidebar:
        st.header("Document Insights")
        
        # Metrics
        with st.container(border=True):
            col1, col2 = st.columns(2)
            col1.metric("Total Words", len(st.session_state.app_state['full_text'].split()))
            col2.metric("Key Points", len(st.session_state.app_state['key_points']))
        
        # Suggested Questions
        if st.session_state.app_state['suggested_questions']:
            with st.container(border=True):
                st.subheader("Suggested Questions")
                for q in st.session_state.app_state['suggested_questions'][:3]:
                    if st.button(q, use_container_width=True, key=f"q_{hash(q)}"):
                        st.session_state.app_state['chat_history'].append(("user", q))
                        st.rerun()
        
        # Chat History
        with st.container(border=True):
            st.subheader("Chat History")
            chat_container = st.container(height=300)
            with chat_container:
                for role, msg in st.session_state.app_state['chat_history'][-5:]:
                    with st.chat_message(role.capitalize()):
                        st.write(msg)

            # Chat Input
            if query := st.chat_input("Ask about the document..."):
                with st.spinner("Analyzing..."):
                    response = "Sample response"  # Add RAG implementation
                    st.session_state.app_state['chat_history'].extend([
                        ("user", query),
                        ("assistant", response)
                    ])
                    st.rerun()

# Main UI
def main():
    # Custom CSS
    st.markdown("""
    <style>
        .main {padding: 2rem;}
        .stSidebar {padding: 1rem;}
        .block-container {padding-top: 2rem;}
        .stButton>button {border-radius: 8px;}
        .stDownloadButton>button {width: 100%;}
        .chat-message {padding: 1rem; margin: 0.5rem 0; border-radius: 8px;}
    </style>
    """, unsafe_allow_html=True)

    # Main Header
    st.title("📘 DocAnalyzer Pro")
    st.markdown("AI-powered Document Analysis and Summarization")

    # Layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # File Processing
        with st.container(border=True):
            st.subheader("Document Upload")
            uploaded_file = st.file_uploader(
                "Choose a file", 
                type=["pdf", "txt", "docx"],
                label_visibility="collapsed"
            )
            
            if uploaded_file and not st.session_state.app_state['document_processed']:
                if st.button("Process Document", type="primary"):
                    with st.status("Processing...", expanded=True) as status:
                        text = safe_extract_text(uploaded_file)
                        if text:
                            chunks = chunk_text(text)
                            st.session_state.app_state.update({
                                'full_text': text,
                                'text_chunks': chunks,
                                'summaries': {'document': map_reduce_summary(chunks)},
                                'key_points': generate_key_points(text),
                                'document_processed': True
                            })
                            status.update(label="Processing Complete!", state="complete")

        # Results Display
        if st.session_state.app_state['document_processed']:
            with st.container():
                # Summary Section
                with st.expander("Document Summary", expanded=True):
                    st.write(st.session_state.app_state['summaries']['document'])
                
                # Key Points
                with st.expander("Key Points", expanded=True):
                    for point in st.session_state.app_state['key_points']:
                        st.markdown(f"- {point}")
                
                # Report Generation
                st.divider()
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Generate PDF Report", use_container_width=True):
                        # PDF generation logic
                        pass
                with col_b:
                    if st.session_state.app_state.get('pdf_buffer'):
                        st.download_button(
                            "Download Report",
                            data=st.session_state.app_state['pdf_buffer'].getvalue(),
                            file_name="report.pdf",
                            use_container_width=True
                        )
    
    with col2:
        sidebar_content()

    # Error Handling
    if st.session_state.app_state['processing_error']:
        st.error(st.session_state.app_state['processing_error'])

if __name__ == "__main__":
    main()