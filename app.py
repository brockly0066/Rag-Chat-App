import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import pandas as pd
import os

# --- Page Config ---
st.set_page_config(
    page_title="Chat with Your Data | RAG System",
    page_icon="🧠",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    
    .main { background-color: #0a0a0f; }
    
    .stApp { background-color: #0a0a0f; color: #f0f0f5; }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6c63ff, #00d4aa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero-sub {
        color: #8888a8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .chat-user {
        background: #1a1a24;
        border-left: 3px solid #6c63ff;
        padding: 1rem 1.25rem;
        border-radius: 0 12px 12px 0;
        margin: 0.75rem 0;
        color: #f0f0f5;
    }
    
    .chat-ai {
        background: #13131a;
        border-left: 3px solid #00d4aa;
        padding: 1rem 1.25rem;
        border-radius: 0 12px 12px 0;
        margin: 0.75rem 0;
        color: #f0f0f5;
    }
    
    .source-badge {
        background: rgba(108,99,255,0.15);
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        color: #6c63ff;
        margin-right: 0.4rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6c63ff, #8b84ff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(108,99,255,0.4);
    }

    .stTextInput > div > div > input {
        background: #1a1a24;
        border: 1px solid #2a2a3a;
        border-radius: 8px;
        color: #f0f0f5;
    }

    .stFileUploader {
        background: #13131a;
        border: 1px dashed #2a2a3a;
        border-radius: 12px;
        padding: 1rem;
    }

    div[data-testid="stSidebar"] {
        background-color: #13131a;
        border-right: 1px solid #2a2a3a;
    }

    .stat-box {
        background: #1a1a24;
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Gemini Setup ---
def setup_gemini(api_key):
    genai.configure(api_key=api_key.strip())
    return genai.GenerativeModel("gemini-2.5-flash")

# --- Extract Text from PDF ---
def extract_pdf_text(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text if text.strip() else "No readable text found in this PDF."
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# --- Extract Text from Excel/CSV ---
def extract_table_text(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
            df = df.fillna("").astype(str)
            return df.to_string(index=False), df
        else:
            # Read ALL sheets
            all_sheets = pd.read_excel(file, sheet_name=None)
            full_text = ""
            for sheet_name, df in all_sheets.items():
                df = df.fillna("").astype(str)
                full_text += f"\n\n=== Sheet: {sheet_name} ===\n"
                full_text += df.to_string(index=False)
            return full_text, None
    except Exception as e:
        return f"Error reading file: {str(e)}", None

# --- Simple Chunking ---
def chunk_text(text, chunk_size=3000, overlap=200):
    if not text or not text.strip():
        return ["No content available."]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- Find Relevant Chunks (keyword-based RAG) ---
def find_relevant_chunks(query, chunks, top_k=4):
    if not chunks:
        return ["No document content available."]
    query_words = set(query.lower().split())
    scored = []
    for i, chunk in enumerate(chunks):
        if chunk and isinstance(chunk, str):
            chunk_words = set(chunk.lower().split())
            score = len(query_words & chunk_words)
            scored.append((score, i, chunk))
    scored.sort(reverse=True)
    return [chunk for _, _, chunk in scored[:top_k]] if scored else chunks[:top_k]

# --- Ask Gemini ---
def ask_gemini(model, question, context, chat_history, sheet_info=""):
    history_text = ""
    for msg in chat_history[-4:]:
        role = str(msg.get('role', '')).upper()
        content = str(msg.get('content', ''))
        history_text += f"{role}: {content}\n"

    prompt = f"""You are a helpful AI assistant. The user has uploaded the following documents: {", ".join(st.session_state.doc_names)}.
{f"IMPORTANT FILE STRUCTURE INFO: {sheet_info}" if sheet_info else ""}

Answer the user's question based on ALL the document content provided below.
- Each chunk is tagged with its source file in [Source: filename] format
- If the user doesn't specify a file, search across ALL documents
- If the answer requires calculation (totals, averages, percentages), perform it using numbers from the context
- If the answer is truly not in any document, say "I couldn't find that in the uploaded documents"

DOCUMENT CONTENT:
{context}

CHAT HISTORY:
{history_text}

USER QUESTION: {question}

Give a clear, helpful answer. Show calculation working if needed:"""

    response = model.generate_content(prompt)
    return response.text

# =====================
# MAIN APP
# =====================

# --- Load API Key from Streamlit Secrets ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = None

# --- Sidebar ---
with st.sidebar:
    if not api_key:
        st.warning("⚠️ API key not configured.")
    else:
        st.success("✅ Gemini AI Connected")

    st.markdown("---")
    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or Excel files",
        type=["pdf", "xlsx", "xls", "csv"],
        accept_multiple_files=True,
        help="Upload one or more documents to chat with"
    )

    if uploaded_files:
        st.markdown("**Uploaded files:**")
        for f in uploaded_files:
            st.markdown(f"✅ `{f.name}`")

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chunks = []
        st.session_state.doc_names = []
        st.session_state.sheet_info = ""
        st.rerun()

    st.markdown("---")
    st.markdown("**Built by** [Sreevardhan](https://mrvardhan006.github.io)")
    st.markdown("**Stack:** Gemini AI · Python · Streamlit")

# --- Main Area ---
st.markdown('<div class="hero-title">🧠 Chat with Your Data</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload PDFs or Excel files and ask questions — powered by Google Gemini AI</div>', unsafe_allow_html=True)

# --- Init session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []
if "sheet_info" not in st.session_state:
    st.session_state.sheet_info = ""

# --- Process uploaded files ---
if uploaded_files and api_key:
    current_names = [f.name for f in uploaded_files]
    if current_names != st.session_state.doc_names:
        with st.spinner("📖 Reading and processing your documents..."):
            all_chunks = []
            for file in uploaded_files:
                if file.name.endswith(".pdf"):
                    text = extract_pdf_text(file)
                    chunks = chunk_text(text)
                    tagged = [f"[Source: {file.name}]\n{c}" for c in chunks]
                    all_chunks.extend(tagged)
                else:
                    # Read ALL sheets and tag each one
                    try:
                        all_sheets = pd.read_excel(file, sheet_name=None)
                        sheet_names = list(all_sheets.keys())
                        st.session_state.sheet_info += f"\nFile '{file.name}' has exactly {len(sheet_names)} sheets: {', '.join(sheet_names)}."
                        summary = f"[Source: {file.name}]\nThis Excel file contains exactly {len(sheet_names)} sheets: {', '.join(sheet_names)}\n"
                        all_chunks.append(summary)
                        for sheet_name, df in all_sheets.items():
                            df = df.fillna("").astype(str)
                            sheet_text = f"[Source: {file.name}] [Sheet: {sheet_name}]\n{df.to_string(index=False)}"
                            chunks = chunk_text(sheet_text)
                            all_chunks.extend(chunks)
                    except Exception as e:
                        text, _ = extract_table_text(file)
                        chunks = chunk_text(text)
                        tagged = [f"[Source: {file.name}]\n{c}" for c in chunks]
                        all_chunks.extend(tagged)

            st.session_state.chunks = all_chunks
            st.session_state.doc_names = current_names
        st.success(f"✅ {len(uploaded_files)} document(s) processed! {len(all_chunks)} text chunks ready.")

# --- Stats row ---
if st.session_state.chunks:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Documents", len(st.session_state.doc_names))
    with col2:
        st.metric("🧩 Text Chunks", len(st.session_state.chunks))
    with col3:
        st.metric("💬 Messages", len(st.session_state.messages))

st.markdown("---")

# --- Chat History ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">👤 <strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-ai">🤖 <strong>AI:</strong> {msg["content"]}</div>', unsafe_allow_html=True)

# --- Input ---
if not api_key:
    st.info("👈 Please enter your **Gemini API Key** in the sidebar to get started.")
elif not uploaded_files:
    st.info("👈 Please **upload at least one PDF or Excel file** in the sidebar.")
else:
    question = st.chat_input("Ask anything about your documents...")
    if question and question.strip():
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("🤔 Thinking..."):
            try:
                api_key_clean = str(api_key).strip()
                model = setup_gemini(api_key_clean)
                relevant = st.session_state.chunks
                # Always include the first chunk (summary) in context
                first_chunk = st.session_state.chunks[0] if st.session_state.chunks else ""
                if first_chunk not in relevant:
                    relevant = [first_chunk] + relevant
                context = "\n\n---\n\n".join([str(c) for c in relevant if c])
                answer = ask_gemini(model, question, context, st.session_state.messages, st.session_state.sheet_info)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
