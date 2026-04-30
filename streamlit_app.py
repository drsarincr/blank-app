import streamlit as st
import os
import subprocess
import threading
import time
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
# 1. TRY START OLLAMA (BEST EFFORT)
# =========================================
def start_ollama():
    try:
        subprocess.Popen(["ollama", "serve"],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        time.sleep(3)
    except:
        pass

def check_ollama():
    try:
        requests.get("http://localhost:11434", timeout=2)
        return True
    except:
        return False

start_ollama()


# =========================================
# 2. UI
# =========================================
st.title("🤖 HR Chatbot")

if not check_ollama():
    st.error("❌ Ollama not running. This app requires local execution.")
    st.stop()


# =========================================
# 3. DATA PATH (AUTO)
# =========================================
DATA_PATH = "HR Policy"

if not os.path.exists(DATA_PATH):
    st.error("❌ HR Policy folder not found")
    st.stop()

st.success(f"📂 Using: {DATA_PATH}")


# =========================================
# 4. LOAD DOCUMENTS
# =========================================
@st.cache_resource
def load_data(path):

    documents = []

    def add_meta(docs, tag):
        for d in docs:
            d.metadata["source"] = d.metadata.get("source", tag)
        return docs

    try:
        docs = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader).load()
        documents.extend(add_meta(docs, "txt"))
    except: pass

    try:
        docs = DirectoryLoader(path, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
        documents.extend(add_meta(docs, "docx"))
    except: pass

    try:
        docs = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyMuPDFLoader).load()
        documents.extend(add_meta(docs, "pdf"))
    except: pass

    if not documents:
        from langchain_core.documents import Document
        documents = [Document(
            page_content="Employees get 20 days leave. Notice period is 30 days.",
            metadata={"source": "demo"}
        )]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


# =========================================
# 5. BUILD CHAIN
# =========================================
def build_chain(vectorstore):

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = OllamaLLM(model="llama3.2")

    prompt = PromptTemplate(
        template="""
You are an HR assistant.

Answer ONLY from context.
If not found, say: "Not found in HR policy."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )


# =========================================
# 6. INIT
# =========================================
with st.spinner("📚 Loading data..."):
    vectorstore = load_data(DATA_PATH)

qa_chain = build_chain(vectorstore)


# =========================================
# 7. CHAT UI
# =========================================
query = st.text_input("Ask HR question:")

if query:
    with st.spinner("🤖 Thinking..."):
        res = qa_chain.invoke({"query": query})

    st.write("### 🧠 Answer")
    st.write(res["result"])

    st.write("### 📄 Sources")
    for d in res["source_documents"]:
        st.write("-", d.metadata.get("source"))
