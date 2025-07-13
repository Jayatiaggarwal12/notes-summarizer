# ========== 1. Set Page Config ==========
import streamlit as st
st.set_page_config(
    page_title="Notes Summarizer",
    page_icon="📘",
    layout="wide"
)

# ========== 2. Other Imports ==========
import os
import nltk
import numpy as np
import faiss
import fitz
import re
import time
import tiktoken
from io import BytesIO
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import docx

# ========== 3. Initialize Environment ==========
nltk.download(['punkt'])
load_dotenv()

# ========== 4. Configuration ==========
class Config:
    # Using a smaller, faster model for better performance
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_LLM_MODEL = "llama3-8b-8192"  # Faster model
    PREMIUM_LLM_MODEL = "llama3-70b-8192"  # Higher quality but slower
    
    # Optimized chunking parameters
    CHUNK_SIZE = 1500  # Increased for fewer API calls
    CHUNK_OVERLAP = 150
    
    # RAG parameters
    VECTOR_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2
    TOP_K = 3
    
    # Rate limiting
    MAX_TOKENS_PER_REQUEST = 1500
    MAX_REQUESTS_PER_MINUTE = 50
    REQUEST_DELAY = 0.5  # Reduced delay
    
    # Processing options
    MAX_WORKERS = 3  # For parallel processing

# ========== 5. Session State ==========
def init_session_state():
    return {
        'chat_history': [],
        'document_processed': False,
        'text_chunks': [],
        'chunk_embeddings': None,
        'faiss_index': None,
        'full_text': "",
        'section_summaries': {},  # Changed to store section-wise summaries
        'key_points': [],
        'exam_questions': [],  # Added for storing exam questions
        'suggested_questions': [],
        'pdf_buffer': None,
        'processing_error': None,
        'document_structure': {},  # Added to store document structure
        'processing_progress': 0,
        'use_premium_model': False  # Toggle for model selection
    }

if 'app_state' not in st.session_state:
    st.session_state.app_state = init_session_state()

# ========== 6. Load Models ==========
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer(Config.EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Embedding model loading failed: {str(e)}")
        return None

@st.cache_resource
def get_llm(premium=False):
    try:
        model_name = Config.PREMIUM_LLM_MODEL if premium else Config.DEFAULT_LLM_MODEL
        return ChatGroq(
            model_name=model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3
        )
    except Exception as e:
        st.error(f"LLM loading failed: {str(e)}")
        return None

embedding_model = load_embedding_model()

# ========== 7. Text Processing Utilities ==========
def clean_heading(heading):
    """Clean chapter or section prefixes from headings"""
    # Remove common chapter/section prefixes
    heading = re.sub(r'^(?:CHAPTER|Chapter|SECTION|Section)\s+[\dIVXLC]+[:\.\s]+', '', heading)
    # Remove numbering
    heading = re.sub(r'^\d+(?:\.\d+)*\s+', '', heading)
    # Capitalize first letter if all caps
    if heading.isupper():
        heading = heading.capitalize()
    return heading.strip()

def extract_sections(text: str) -> Dict[str, str]:
    """Extract sections from document based on headings"""
    # Simple regex pattern to detect headings
    heading_patterns = [
        # Chapter or Section indicators
        r'(?:CHAPTER|Chapter|SECTION|Section)\s+[\dIVXLC]+[:\.\s]+(.*?)(?=\n)',
        # Numbered headings
        r'^(?:\d+(?:\.\d+)*)\s+(.*?)(?=\n)',
        # Capitalized headings of 2-6 words
        r'^([A-Z][A-Z\s]{5,50})(?=\n)',
        # Common heading indicators
        r'^(?:Introduction|Conclusion|Abstract|Summary|Background|Methodology|Results|Discussion|References)(?=\n|\s|:)',
    ]
    
    # Find potential headings
    potential_headings = []
    for pattern in heading_patterns:
        potential_headings.extend([(m.start(), m.group()) for m in re.finditer(pattern, text, re.MULTILINE)])
    
    # Sort headings by position in text
    potential_headings.sort()
    
    # Extract sections
    sections = {}
    if not potential_headings:
        # If no headings found, treat as single section
        sections["Main Content"] = text
    else:
        # Process each heading and its content
        for i, (pos, heading) in enumerate(potential_headings):
            next_pos = potential_headings[i+1][0] if i+1 < len(potential_headings) else len(text)
            section_text = text[pos:next_pos].strip()
            
            # Clean heading
            clean_head = clean_heading(heading)
            sections[clean_head] = section_text
    
    return sections

def chunk_text(text: str, max_tokens: int = Config.CHUNK_SIZE) -> List[str]:
    """Split text into chunks based on token count with optimized overlap"""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens - Config.CHUNK_OVERLAP):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk))
        
        # Break if chunk is too small to avoid unnecessary processing
        if len(chunk) < Config.CHUNK_SIZE // 2:
            break
    
    return chunks

def create_chunks_for_rag(text: str) -> List[str]:
    """Create smaller chunks for RAG retrieval"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len
    )
    return text_splitter.split_text(text)

def safe_extract_text(uploaded_file):
    """Extract text from uploaded file with error handling"""
    try:
        file_bytes = uploaded_file.getvalue()
        if uploaded_file.type == "application/pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            return "\n".join([page.get_text() for page in doc])
        elif uploaded_file.type == "text/plain":
            return uploaded_file.getvalue().decode("utf-8")
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(BytesIO(file_bytes))
            return "\n".join([para.text for para in doc.paragraphs])
        return ""
    except Exception as e:
        st.session_state.app_state['processing_error'] = f"Text extraction error: {str(e)}"
        return ""

# ========== 8. RAG Implementation ==========
def build_faiss_index(texts: List[str]):
    """Build FAISS index from text chunks"""
    embeddings = []
    for text in texts:
        embedding = embedding_model.encode(text)
        embeddings.append(embedding)
    
    # Convert to numpy array
    embeddings_np = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    index = faiss.IndexFlatL2(Config.VECTOR_DIMENSION)
    index.add(embeddings_np)
    
    return index, embeddings_np

def get_relevant_chunks(question: str, top_k: int = Config.TOP_K) -> List[str]:
    """Retrieve relevant chunks for a question"""
    # Get question embedding
    question_embedding = embedding_model.encode(question)
    question_embedding = np.array([question_embedding]).astype('float32')
    
    # Search in FAISS index
    app_state = st.session_state.app_state
    if app_state['faiss_index'] is None or len(app_state['text_chunks']) == 0:
        return []
    
    distances, indices = app_state['faiss_index'].search(question_embedding, top_k)
    
    # Get relevant chunks
    relevant_chunks = [app_state['text_chunks'][i] for i in indices[0]]
    return relevant_chunks

def answer_question(question: str, use_premium: bool = False) -> str:
    """Answer a question using RAG"""
    # Get relevant chunks
    relevant_chunks = get_relevant_chunks(question)
    
    if not relevant_chunks:
        return "I don't have enough information to answer that question based on the document."
    
    # Combine chunks
    context = "\n\n".join(relevant_chunks)
    
    # Get LLM
    llm = get_llm(premium=use_premium)
    
    # Create prompt
    prompt = PromptTemplate.from_template("""
    You are a helpful educational assistant answering questions about a document.
    Use the following context from the document to answer the question.
    If the answer is not in the context, say "I don't have that information in the document."
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    ANSWER:
    """)
    
    # Run chain
    try:
        answer = LLMChain(llm=llm, prompt=prompt).run({
            "context": context,
            "question": question
        })
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# ========== 9. AI Processing Functions ==========
@st.cache_data(ttl=3600, show_spinner=False)
def generate_key_points_cached(text: str, use_premium: bool = False) -> List[str]:
    """Cached version of key points generation"""
    return generate_key_points(text, use_premium)
    
def generate_key_points(text: str, use_premium: bool = False) -> List[str]:
    """Generate key points from text using chunked processing"""
    try:
        llm = get_llm(premium=use_premium)
        chunks = chunk_text(text, Config.MAX_TOKENS_PER_REQUEST)
        all_points = []
        
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            prompt = PromptTemplate.from_template("""
            You are an expert educator analyzing study material.
            Extract 3-5 key points from this text that would be IMPORTANT FOR EXAMS.
            Focus on concepts, definitions, and facts that typically appear in exams.
            Format as bullet points starting with "-" and make them concise:
            
            {text}
            
            KEY POINTS:
            """)
            
            try:
                result = LLMChain(llm=llm, prompt=prompt).run({"text": chunk})
                points = [p.strip() for p in result.split('\n') if p.strip().startswith('-')]
                all_points.extend(points)
                
                # Stop if we have enough points
                if len(all_points) >= 10:
                    break
                    
                # Add delay to avoid rate limits
                time.sleep(Config.REQUEST_DELAY)
                    
            except Exception as e:
                st.warning(f"Partial processing error: {str(e)}")
                continue
        
        # Deduplicate and limit points
        unique_points = list(set(all_points))[:10]
        return unique_points if unique_points else ["Could not extract key points"]
        
    except Exception as e:
        return [f"Key points error: {str(e)}"]

@st.cache_data(ttl=3600, show_spinner=False)
def generate_section_summary_cached(section_name: str, text: str, use_premium: bool = False) -> str:
    """Cached version of section summary generation"""
    return generate_section_summary(section_name, text, use_premium)

def generate_section_summary(section_name: str, text: str, use_premium: bool = False) -> str:
    """Generate summary for a specific section"""
    try:
        llm = get_llm(premium=use_premium)
        chunks = chunk_text(text, Config.MAX_TOKENS_PER_REQUEST)
        summaries = []
        
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            prompt = PromptTemplate.from_template("""
            You are an expert study guide creator specializing in concise, informative summaries.
            Write a concise, informative summary of this section.
            Focus on capturing key concepts that would appear in exams.
            
            {text}
            
            SUMMARY:
            """)
            
            try:
                summary = LLMChain(llm=llm, prompt=prompt).run({
                    "text": chunk
                })
                summaries.append(summary)
                
                # Add delay to avoid rate limits
                time.sleep(Config.REQUEST_DELAY)
                
            except Exception as e:
                st.warning(f"Partial summary error: {str(e)}")
                continue
        
        # Combine summaries if needed
        if summaries:
            if len(summaries) == 1:
                return summaries[0]
            
            combine_prompt = PromptTemplate.from_template("""
            Combine these summaries into one final comprehensive summary:
            
            {summaries}
            
            FINAL SUMMARY:
            """)
            
            final_summary = LLMChain(llm=llm, prompt=combine_prompt).run({
                "summaries": "\n".join(summaries)
            })
            return final_summary
        else:
            return "Summary generation failed"
            
    except Exception as e:
        return f"Summary error: {str(e)}"

@st.cache_data(ttl=3600, show_spinner=False)
def generate_exam_questions_cached(text: str, use_premium: bool = False) -> List[str]:
    """Cached version of exam question generation"""
    return generate_exam_questions(text, use_premium)

def generate_exam_questions(text: str, use_premium: bool = False) -> List[str]:
    """Generate potential exam questions from content"""
    try:
        llm = get_llm(premium=use_premium)
        chunks = chunk_text(text, Config.MAX_TOKENS_PER_REQUEST)
        all_questions = []
        
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            prompt = PromptTemplate.from_template("""
            You are an experienced professor creating exam questions.
            Generate 3-5 potential exam questions based on this material.
            Include a mix of factual recall, concept application, and critical thinking questions.
            Format each question on a new line starting with "Q:".
            
            {text}
            
            EXAM QUESTIONS:
            """)
            
            try:
                result = LLMChain(llm=llm, prompt=prompt).run({"text": chunk})
                questions = [q.strip().replace("Q:", "").strip() for q in result.split('\n') if "Q:" in q]
                all_questions.extend(questions)
                
                # Add delay to avoid rate limits
                time.sleep(Config.REQUEST_DELAY)
                    
            except Exception as e:
                st.warning(f"Partial question generation error: {str(e)}")
                continue
        
        # Deduplicate and limit questions
        unique_questions = list(set(all_questions))[:8]
        return unique_questions if unique_questions else ["Could not generate exam questions"]
        
    except Exception as e:
        return [f"Question generation error: {str(e)}"]

def generate_report_pdf(app_state: Dict[str, Any]) -> BytesIO:
    """Generate a PDF report from processed document data"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Content elements
    elements = []
    
    # Title
    title_style = styles["Title"]
    elements.append(Paragraph("Document Analysis Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Key Points
    heading_style = styles["Heading1"]
    elements.append(Paragraph("Key Points for Exam", heading_style))
    elements.append(Spacer(1, 6))
    
    normal_style = styles["Normal"]
    for point in app_state['key_points']:
        elements.append(Paragraph(f"• {point}", normal_style))
        elements.append(Spacer(1, 3))
    
    elements.append(Spacer(1, 12))
    
    # Section Summaries
    elements.append(Paragraph("Section Summaries", heading_style))
    elements.append(Spacer(1, 6))
    
    section_style = styles["Heading2"]
    for section, summary in app_state['section_summaries'].items():
        elements.append(Paragraph(section, section_style))
        elements.append(Spacer(1, 3))
        elements.append(Paragraph(summary, normal_style))
        elements.append(Spacer(1, 10))
    
    # Potential Exam Questions
    elements.append(PageBreak())
    elements.append(Paragraph("Potential Exam Questions", heading_style))
    elements.append(Spacer(1, 6))
    
    for i, question in enumerate(app_state['exam_questions'], 1):
        elements.append(Paragraph(f"Question {i}: {question}", normal_style))
        elements.append(Spacer(1, 6))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ========== 10. Parallel Processing ==========
def process_sections(sections, use_premium):
    """Process sections in parallel"""
    section_summaries = {}
    progress_per_section = 90 / len(sections)
    
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        future_to_section = {
            executor.submit(generate_section_summary_cached, name, content, use_premium): name
            for name, content in sections.items()
        }
        
        for i, future in enumerate(future_to_section):
            name = future_to_section[future]
            try:
                section_summaries[name] = future.result()
                # Update progress
                st.session_state.app_state['processing_progress'] = 10 + (i + 1) * progress_per_section
            except Exception as e:
                section_summaries[name] = f"Error processing section: {str(e)}"
    
    return section_summaries

# ========== 11. Main UI ==========
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
        .highlight {background-color: #ffd70030; padding: 0.2rem; border-radius: 3px;}
        .progress-bar {height: 6px; background-color: #f0f2f6; border-radius: 3px; margin-bottom: 10px;}
        .progress-bar-fill {height: 100%; background-color: #3c9ef6; border-radius: 3px; transition: width 0.5s;}
        .stRadio [role=radiogroup]{flex-direction:row;}
    </style>
    """, unsafe_allow_html=True)

    # Main Header
    st.title("📘 Notes Summarizer")
    st.markdown("AI-powered Document Analysis and Exam Preparation")

    # Layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main content area
        tabs = st.tabs(["📄 Document", "📝 Summary", "❓ Exam Questions", "💬 Chat"])
        
        # Document Tab
        with tabs[0]:
            # File Processing
            with st.container(border=True):
                st.subheader("Document Upload")
                
                # Model selection
                model_options = st.radio(
                    "Processing Speed:",
                    ["Fast (8B model)", "High Quality (70B model)"],
                    horizontal=True,
                    index=0
                )
                st.session_state.app_state['use_premium_model'] = model_options == "High Quality (70B model)"
                
                uploaded_file = st.file_uploader(
                    "Choose a file", 
                    type=["pdf", "txt", "docx"]
                )
                
                if uploaded_file and not st.session_state.app_state['document_processed']:
                    if st.button("Process Document", type="primary", use_container_width=True):
                        # Reset progress
                        st.session_state.app_state['processing_progress'] = 0
                        
                        with st.status("Processing document...", expanded=True) as status:
                            # Extract text
                            st.write("Extracting text...")
                            text = safe_extract_text(uploaded_file)
                            st.session_state.app_state['processing_progress'] = 5
                            
                            if text:
                                # Extract document structure
                                st.write("Analyzing document structure...")
                                sections = extract_sections(text)
                                st.session_state.app_state['document_structure'] = sections
                                st.session_state.app_state['processing_progress'] = 10
                                
                                # Process sections in parallel
                                st.write("Generating section summaries...")
                                use_premium = st.session_state.app_state['use_premium_model']
                                section_summaries = process_sections(sections, use_premium)
                                st.session_state.app_state['processing_progress'] = 70
                                
                                # Generate key points and exam questions
                                st.write("Creating exam preparation materials...")
                                key_points = generate_key_points_cached(text, use_premium)
                                exam_questions = generate_exam_questions_cached(text, use_premium)
                                st.session_state.app_state['processing_progress'] = 90
                                
                                # Build RAG index
                                st.write("Building search index for Q&A...")
                                text_chunks = create_chunks_for_rag(text)
                                faiss_index, chunk_embeddings = build_faiss_index(text_chunks)
                                
                                # Update session state
                                st.session_state.app_state.update({
                                    'full_text': text,
                                    'section_summaries': section_summaries,
                                    'key_points': key_points,
                                    'exam_questions': exam_questions,
                                    'text_chunks': text_chunks,
                                    'faiss_index': faiss_index,
                                    'chunk_embeddings': chunk_embeddings,
                                    'document_processed': True,
                                    'processing_progress': 100
                                })
                                
                                # Generate PDF report
                                pdf_buffer = generate_report_pdf(st.session_state.app_state)
                                st.session_state.app_state['pdf_buffer'] = pdf_buffer
                                
                                status.update(label="Processing Complete!", state="complete")
            
            # Show document information if processed
            if st.session_state.app_state['document_processed']:
                with st.container(border=True):
                    st.subheader("Document Information")
                    st.info(f"Document processed successfully. {len(st.session_state.app_state['section_summaries'])} sections identified with {len(st.session_state.app_state['full_text'].split())} total words.")
                    st.success("Use the tabs above to view summaries, exam questions, or chat with your document!")
        
        # Summary Tab
        with tabs[1]:
            if st.session_state.app_state['document_processed']:
                st.subheader("Document Summaries")
                
                for section_name, summary in st.session_state.app_state['section_summaries'].items():
                    with st.expander(section_name, expanded=True):
                        st.markdown(summary)
            else:
                st.info("Please upload and process a document to view summaries.")
        
        # Exam Questions Tab
        with tabs[2]:
            if st.session_state.app_state['document_processed']:
                st.subheader("Potential Exam Questions")
                
                for i, question in enumerate(st.session_state.app_state['exam_questions'], 1):
                    with st.container(border=True):
                        st.markdown(f"**Question {i}:** {question}")
                
                st.subheader("Key Points to Remember")
                for point in st.session_state.app_state['key_points']:
                    st.markdown(f"- {point}")
            else:
                st.info("Please upload and process a document to view exam questions.")
        
        # Chat Tab
        with tabs[3]:
            if st.session_state.app_state['document_processed']:
                st.subheader("Chat with Your Document")
                
                # Display chat history
                for role, msg in st.session_state.app_state['chat_history']:
                    with st.chat_message(role):
                        st.write(msg)
                
                # Chat input
                if query := st.chat_input("Ask about the document..."):
                    # Add user message to history
                    st.session_state.app_state['chat_history'].append(("user", query))
                    
                    # Display user message (since we'll rerun after getting response)
                    with st.chat_message("user"):
                        st.write(query)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            # Get response using RAG
                            use_premium = st.session_state.app_state['use_premium_model']
                            response = answer_question(query, use_premium)
                            st.write(response)
                    
                    # Add assistant response to history
                    st.session_state.app_state['chat_history'].append(("assistant", response))
            else:
                st.info("Please upload and process a document to chat with it.")

    with col2:
        # Sidebar Content
        with st.sidebar:
            st.header("Document Insights")
            
            # Processing progress indicator
            if not st.session_state.app_state['document_processed'] and st.session_state.app_state['processing_progress'] > 0:
                progress = st.session_state.app_state['processing_progress']
                st.markdown(f"""
                <div class="progress-bar">
                    <div class="progress-bar-fill" style="width: {progress}%;"></div>
                </div>
                Processing: {progress:.0f}%
                """, unsafe_allow_html=True)
            
            # Document metrics
            if st.session_state.app_state['document_processed']:
                with st.container(border=True):
                    col1, col2 = st.columns(2)
                    word_count = len(st.session_state.app_state['full_text'].split())
                    col1.metric("Total Words", f"{word_count:,}")
                    col2.metric("Key Points", len(st.session_state.app_state['key_points']))
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Sections", len(st.session_state.app_state['section_summaries']))
                    col2.metric("Questions", len(st.session_state.app_state['exam_questions']))
                
                # Download report
                if st.session_state.app_state['pdf_buffer']:
                    st.download_button(
                        label="📥 Download Study Guide",
                        data=st.session_state.app_state['pdf_buffer'],
                        file_name="study_guide.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                
                # Exam practice
                with st.container(border=True):
                    st.subheader("📝 Exam Practice")
                    if st.session_state.app_state['exam_questions']:
                        # Randomly select a question to practice with
                        import random
                        question_idx = random.randint(0, len(st.session_state.app_state['exam_questions'])-1)
                        practice_question = st.session_state.app_state['exam_questions'][question_idx]
                        
                        st.markdown(f"**Q:** {practice_question}")
                        
                        # Simple way to practice answering
                        user_answer = st.text_area("Your Answer", height=100)
                        if st.button("Check Answer", use_container_width=True):
                            st.success("Practice response submitted! Review your knowledge with the study guide.")
            
            # Error Handling
            if st.session_state.app_state['processing_error']:
                st.error(st.session_state.app_state['processing_error'])

# ========== 12. Run App ==========
if __name__ == "__main__":
    main()
