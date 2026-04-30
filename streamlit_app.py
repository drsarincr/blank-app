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
# 1. CHECK OLLAMA SERVER
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
st.set_page_config(page_title="HR Chatbot", layout="centered")
st.title("🤖 HR Chatbot (Local - Ollama/Qwen)")


if not check_ollama():
    st.error("❌ Ollama is not running")
    st.code("ollama serve", language="bash")
    st.info("👉 Run the above command in terminal, then refresh")
    st.stop()


# =========================================
# 3. MODEL SELECTION
# =========================================
model_name = st.selectbox(
    "Select Model",
    ["qwen:7b", "llama3", "mistral"]
)


# =========================================
# 4. DATA PATH
# =========================================
DATA_PATH = "HR Policy"

if not os.path.exists(DATA_PATH):
    st.error("❌ 'HR Policy' folder not found")
    st.stop()

st.success(f"📂 Using: {DATA_PATH}")


# =========================================
# 5. LOAD DOCUMENTS (CACHED)
# =========================================
@st.cache_resource
def load_vectorstore(path):

    documents = []

    def add_meta(docs, tag):
        for d in docs:
            d.metadata["source"] = d.metadata.get("source", tag)
        return docs

    try:
        docs = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader).load()
        documents.extend(add_meta(docs, "txt"))
    except:
        pass

    try:
        docs = DirectoryLoader(path, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
        documents.extend(add_meta(docs, "docx"))
    except:
        pass

    try:
        docs = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyMuPDFLoader).load()
        documents.extend(add_meta(docs, "pdf_fast"))
    except:
        pass

    try:
        docs = DirectoryLoader(path, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader).load()
        documents.extend(add_meta(docs, "pdf_ocr"))
    except:
        pass

    # fallback
    if len(documents) == 0:
        from langchain_core.documents import Document
        documents = [Document(
            page_content="Employees get 20 days leave. Notice period is 30 days.",
            metadata={"source": "demo"}
        )]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)
    docs = [d for d in docs if d.page_content.strip() != ""]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embeddings)


# =========================================
# 6. BUILD QA CHAIN
# =========================================
def build_chain(vectorstore, model):

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = OllamaLLM(model=model)

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

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )


# =========================================
# 7. INITIALIZE
# =========================================
with st.spinner("📚 Indexing documents..."):
    vectorstore = load_vectorstore(DATA_PATH)

qa_chain = build_chain(vectorstore, model_name)


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
            st.error("❌ Model failed to respond")
            st.exception(e)
