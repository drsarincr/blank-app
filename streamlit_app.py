import streamlit as st
import os
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
# 1. Load HR Documents
# =========================================
DATA_PATH = "HR_Policy"  # folder in your repo with HR docs

documents = []
def add_meta(docs, tag):
    for d in docs:
        d.metadata["source"] = d.metadata.get("source", tag)
    return docs

try:
    docs = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader).load()
    documents.extend(add_meta(docs, "txt"))
except: pass

try:
    docs = DirectoryLoader(DATA_PATH, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
    documents.extend(add_meta(docs, "docx"))
except: pass

try:
    docs = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyMuPDFLoader).load()
    documents.extend(add_meta(docs, "pdf_fast"))
except: pass

try:
    docs = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader).load()
    documents.extend(add_meta(docs, "pdf_ocr"))
except: pass

# Fallback demo data
if len(documents) == 0:
    from langchain_core.documents import Document
    documents = [Document(
        page_content="Employees get 20 days leave. Notice period is 30 days.",
        metadata={"source": "demo"}
    )]

# =========================================
# 2. Split + Embed
# =========================================
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)
docs = [d for d in docs if d.page_content.strip() != ""]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =========================================
# 3. LLM + Prompt
# =========================================
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
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# =========================================
# 4. Streamlit UI
# =========================================
st.title("🤖 HR Chatbot")

user_input = st.text_input("Ask a question about HR policy:")

if user_input:
    res = qa_chain.invoke({"query": user_input})
    st.markdown("### 🧠 Answer")
    st.write(res["result"])

    st.markdown("### 📄 Sources")
    for d in res["source_documents"]:
        st.write("-", d.metadata.get("source"))
