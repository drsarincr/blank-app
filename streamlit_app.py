import streamlit as st
import os
import subprocess
import time
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyMuPDFLoader,
    Docx2txtLoader, UnstructuredPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

# Page config
st.set_page_config(page_title="HR Policy Chatbot", page_icon="🤖", layout="wide")

# Title
st.title("🤖 HR Policy Chatbot")
st.markdown("Ask questions about HR policies and get instant answers!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("This chatbot uses Ollama (llama3.2) and RAG to answer HR policy questions.")
    
    # Ollama server status
    ollama_status = st.empty()

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to check if Ollama is running
def check_ollama():
    try:
        import ollama
        ollama.list()
        return True
    except:
        return False

# Function to initialize the RAG system
@st.cache_resource
def initialize_rag():
    DATA_PATH = "HR Policy"  # Your folder in GitHub
    
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Path not found: {DATA_PATH}")
        return None
    
    # Load documents
    documents = []
    
    def add_meta(docs, tag):
        for d in docs:
            d.metadata["source"] = d.metadata.get("source", tag)
        return docs
    
    # Load TXT files
    try:
        docs = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader).load()
        documents.extend(add_meta(docs, "txt"))
    except:
        pass
    
    # Load DOCX files
    try:
        docs = DirectoryLoader(DATA_PATH, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
        documents.extend(add_meta(docs, "docx"))
    except:
        pass
    
    # Load PDF files (fast)
    try:
        docs = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyMuPDFLoader).load()
        documents.extend(add_meta(docs, "pdf_fast"))
    except:
        pass
    
    # Load PDF files with OCR
    try:
        docs = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader).load()
        documents.extend(add_meta(docs, "pdf_ocr"))
    except:
        pass
    
    if len(documents) == 0:
        st.warning("⚠️ No documents loaded, using demo data")
        documents = [Document(
            page_content="Employees get 20 days leave. Notice period is 30 days.",
            metadata={"source": "demo"}
        )]
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    docs = [d for d in docs if d.page_content and d.page_content.strip() != ""]
    
    st.sidebar.success(f"✅ Loaded {len(docs)} document chunks")
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Initialize LLM
    llm = OllamaLLM(model="llama3.2")
    
    # Create prompt
    prompt = PromptTemplate(
        template="""
You are an HR assistant.
Answer ONLY from the context.
If not found, say: "Not found in HR policy."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain

# Check Ollama status
if check_ollama():
    ollama_status.success("✅ Ollama is running")
    
    # Initialize RAG if not done
    if st.session_state.qa_chain is None:
        with st.spinner("🔄 Initializing RAG system..."):
            st.session_state.qa_chain = initialize_rag()
else:
    ollama_status.error("❌ Ollama server not running. Please start Ollama first.")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📄 View Sources"):
                for source in message["sources"]:
                    st.text(f"- {source}")

# Chat input
if prompt := st.chat_input("Ask about HR policies..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    if st.session_state.qa_chain:
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.qa_chain.invoke({"query": prompt})
                answer = response["result"]
                sources = [d.metadata.get("source", "Unknown") for d in response["source_documents"]]
                
                st.markdown(answer)
                with st.expander("📄 View Sources"):
                    for source in sources:
                        st.text(f"- {source}")
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
