import streamlit as st
import os
import time
import tempfile
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(page_title="NIM AI Insight Agent", page_icon=":material/auto_awesome:")
st.title(":material/auto_awesome: NIM AI Insight Agent")
st.markdown("### Powered by Llama 3.3 70B & NVIDIA NIM")

if not api_key:
    st.error("NVIDIA_API_KEY not found. Please check your .env file.")
    st.stop()

llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", nvidia_api_key=api_key)
embedder = NVIDIAEmbeddings(model="nvidia/llama-3.2-nv-embedqa-1b-v2", model_type="passage")
query_embedder = NVIDIAEmbeddings(model="nvidia/llama-3.2-nv-embedqa-1b-v2", model_type="query")

def run_nvidia_rag(pdf_path, question, original_filename, k_value):
    start_time = time.time()
    db_path = f"faiss_index_{original_filename.replace(' ', '_')}" 

    if os.path.exists(db_path):
        vectorstore = FAISS.load_local(db_path, query_embedder, allow_dangerous_deserialization=True)
        total_chunks = vectorstore.index.ntotal 
    else:
        with st.spinner("Processing PDF (First-time ingestion)..."):
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)
            total_chunks = len(splits)
            
            vectorstore = FAISS.from_documents(documents=splits, embedding=embedder)
            vectorstore.save_local(db_path)

    query_vector = query_embedder.embed_query(question)
    context_docs = vectorstore.similarity_search_by_vector(query_vector, k=k_value)
    
    retrieved_texts = [doc.page_content for doc in context_docs] 
    
    sources = [f"Page {doc.metadata.get('page', 'Unknown') + 1}" for doc in context_docs]
    unique_sources = sorted(list(set(sources)))
    context_text = "\n\n".join(retrieved_texts)
    
    prompt = f"Context: {context_text}\n\nQuestion: {question}"
    response = llm.invoke(prompt)
    
    latency = round(time.time() - start_time, 2)
    return response.content, unique_sources, latency, total_chunks, retrieved_texts

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Ask a question about your document:", placeholder="e.g., What are the key performance metrics?")

st.sidebar.markdown("---")
st.sidebar.subheader(":material/tune: RAG Configuration")

k_chunks = st.sidebar.slider(
    "Context Chunks (k)", 
    min_value=2, 
    max_value=10, 
    value=6, 
    help="Higher 'k' increases accuracy but also increases latency."
)

if uploaded_file and question:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        answer, pages, latency, total_chunks, retrieved_chunks = run_nvidia_rag(tmp_path, question, uploaded_file.name, k_chunks)
        # Displays Results
        st.markdown("---")
        st.subheader(":material/auto_awesome: Agent Insights")
        st.write(answer)
        with st.expander(":material/database: View Retrieved Context Chunks"):
            st.info("These are the specific text segments retrieved from the vector database to ground the AI's response.")
            for i, text in enumerate(retrieved_chunks):
                st.markdown(f"**Chunk {i+1}**")
                st.caption(text)
                st.markdown("---")
        # Displays Performance Metrics
        st.sidebar.markdown("---")
        st.sidebar.subheader(":material/monitoring: System Metrics")

        st.sidebar.markdown(f"""
    <div style="
        background-color: #1e1e1e; 
        padding: 6px 12px; 
        border-radius: 6px; 
        border-left: 4px solid #76b900;
        margin-bottom: 10px;">
        <p style="color: #76b900; margin: 0; font-weight: bold; font-size: 0.7rem; letter-spacing: 0.5px;">LATENCY</p>
        <h3 style="margin: 0; font-size: 1.2rem; font-family: monospace;">{latency}s</h3>
    </div>
    """, unsafe_allow_html=True)

        st.sidebar.markdown("<br>", unsafe_allow_html=True) 
        st.sidebar.caption("RETRIEVAL")
        st.sidebar.write("`Asymmetric RAG` | `NV-EmbedQA-1B`")
        st.sidebar.caption("DATABASE")
        st.sidebar.write(f"`{total_chunks}` Total Document Chunks")
        st.sidebar.write(f"`{k_chunks}` Context Chunks (k)")
        st.sidebar.write(f":material/source: **Sources:** {', '.join(pages)} ")

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        # Cleans up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

st.sidebar.markdown("---")
if st.sidebar.button(":material/delete: Clear Vector Cache"):
    import shutil
    count = 0
    for folder in os.listdir():
        if folder.startswith("faiss_index_"):
            shutil.rmtree(folder)
            count += 1
    st.sidebar.success(f"Cleared {count} cached index(es)!")
    time.sleep(1)
    st.rerun() 
            
else:
    st.info("Upload a PDF and type a question to get started.")