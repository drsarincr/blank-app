import streamlit as st
import os
import requests

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

# =========================================
# 1. CHECK OLLAMA (NO AUTO-START ❗)
# =========================================
def check_ollama():
    try:
        requests.get("http://localhost:11434", timeout=2)
        return True
    except:
        return False

# =========================================
# 2. UI
# =========================================
st.title("🤖 HR Chatbot (Ollama - Local Only)")

if not check_ollama():
    st.error("❌ Ollama is not running.\n\n👉 Run in terminal:\n\nollama serve")
    st.stop()

# =========================================
# 3. DATA PATH (AUTO)
# =========================================
DATA_PATH = "HR Policy"

if not os.path.exists(DATA_PATH):
    st.error("❌ 'HR Policy' folder not found in project")
    st.stop()
else:
    st.success(f"📂 Using data from: {DATA_PATH}")

# =========================================
# 4. LOAD DOCUMENTS + VECTORSTORE
# =========================================
@st.cache_resource
def load_vectorstore(path):

    documents = []

    def add_meta(docs, tag):
        for d in docs:
            d.metadata["source"] = d.metadata.get("source", tag)
        return docs

    # TXT
    try:
        docs = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader).load()
        documents.extend(add_meta(docs, "txt"))
    except:
        pass

    # DOCX
    try:
        docs = DirectoryLoader(path, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
        documents.extend(add_meta(docs, "docx"))
    except:
        pass

    # PDF fast
    try:
        docs = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyMuPDFLoader).load()
        documents.extend(add_meta(docs, "pdf_fast"))
    except:
        pass

    # PDF OCR fallback
    try:
        docs = DirectoryLoader(path, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader).load()
        documents.extend(add_meta(docs, "pdf_ocr"))
    except:
        pass

    # Fallback if no files found
    if len(documents) == 0:
        from langchain_core.documents import Document
        documents = [Document(
            page_content="Employees get 20 days leave. Notice period is 30 days.",
            metadata={"source": "demo"}
        )]

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)
    docs = [d for d in docs if d.page_content.strip() != ""]

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

# =========================================
# 5. BUILD QA CHAIN (NO CACHE ❗)
# =========================================
def build_chain(vectorstore):

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = OllamaLLM(model="llama3.2")

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

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain

# =========================================
# 6. LOAD SYSTEM
# =========================================
with st.spinner("📚 Loading HR documents..."):
    vectorstore = load_vectorstore(DATA_PATH)

qa_chain = build_chain(vectorstore)

# =========================================
# 7. CHAT UI
# =========================================
user_query = st.text_input("Ask your HR question:")

if user_query:
    with st.spinner("🤖 Thinking..."):
        res = qa_chain.invoke({"query": user_query})

    st.markdown("### 🧠 Answer")
    st.write(res["result"])

    st.markdown("### 📄 Sources")
    for d in res["source_documents"]:
        st.write("-", d.metadata.get("source", "HR Policy"))
