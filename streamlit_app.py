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
# 1. CHECK OLLAMA
# =========================================
def check_ollama():
    try:
        res = requests.get("http://localhost:11434", timeout=2)
        return res.status_code == 200
    except:
        return False


# =========================================
# 2. UI
# =========================================
st.set_page_config(page_title="HR Chatbot", layout="centered")
st.title("🤖 HR Chatbot (Ollama - Local Only)")


# =========================================
# 3. CHECK OLLAMA STATUS
# =========================================
if not check_ollama():
    st.error("❌ Ollama is NOT running")

    st.code("ollama serve", language="bash")

    st.info("👉 Open a terminal and run the above command.\nThen refresh this page.")

    st.stop()


# =========================================
# 4. DATA PATH (AUTO)
# =========================================
DATA_PATH = "HR Policy"

if not os.path.exists(DATA_PATH):
    st.error("❌ 'HR Policy' folder not found in project directory")
    st.stop()

st.success(f"📂 Using data from: {DATA_PATH}")


# =========================================
# 5. LOAD VECTORSTORE (CACHED)
# =========================================
@st.cache_resource
def load_vectorstore(path):

    documents = []

    def add_meta(docs, tag):
        for d in docs:
            d.metadata["source"] = d.metadata.get("source", tag)
        return docs

    # Load TXT
    try:
        docs = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader).load()
        documents.extend(add_meta(docs, "txt"))
    except:
        pass

    # Load DOCX
    try:
        docs = DirectoryLoader(path, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
        documents.extend(add_meta(docs, "docx"))
    except:
        pass

    # Load PDF (fast)
    try:
        docs = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyMuPDFLoader).load()
        documents.extend(add_meta(docs, "pdf_fast"))
    except:
        pass

    # Load PDF OCR fallback
    try:
        docs = DirectoryLoader(path, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader).load()
        documents.extend(add_meta(docs, "pdf_ocr"))
    except:
        pass

    # Fallback if nothing found
    if len(documents) == 0:
        from langchain_core.documents import Document
        documents = [Document(
            page_content="Employees get 20 days leave. Notice period is 30 days.",
            metadata={"source": "demo"}
        )]

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    docs = [d for d in docs if d.page_content.strip() != ""]

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore


# =========================================
# 6. BUILD QA CHAIN (NO CACHE)
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
# 7. LOAD SYSTEM
# =========================================
with st.spinner("📚 Indexing HR documents..."):
    vectorstore = load_vectorstore(DATA_PATH)

qa_chain = build_chain(vectorstore)


# =========================================
# 8. CHAT UI
# =========================================
query = st.text_input("Ask your HR question:")

if query:
    with st.spinner("🤖 Thinking..."):
        try:
            res = qa_chain.invoke({"query": query})

            st.markdown("### 🧠 Answer")
            st.write(res["result"])

            st.markdown("### 📄 Sources")
            for d in res["source_documents"]:
                st.write("-", d.metadata.get("source", "HR Policy"))

        except Exception as e:
            st.error("❌ Failed to get response from Ollama")
            st.exception(e)
