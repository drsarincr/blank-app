import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers

st.title("📄 Simple HR Bot")

# 1. Load Data
@st.cache_resource
def get_data():
    data_path = "HR Policy"
    if not os.path.exists(data_path): return None
    loader = TextLoader(data_path)
    docs = loader.load()
    # We take only the first 200 characters to skip your list of numbers
    docs[0].page_content = docs[0].page_content[:200]
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings).as_retriever()

# 2. Load the TINIEST model possible (Qwen 0.5B)
@st.cache_resource
def get_llm():
    return CTransformers(
        model="Qwen/Qwen2-0.5B-Instruct-GGUF",
        model_file="qwen2-0.5b-instruct-q4_k_m.gguf",
        model_type="gpt2"
    )

retriever = get_data()
llm = get_llm()

if query := st.chat_input("Question:"):
    st.chat_message("user").write(query)
    context = retriever.get_relevant_documents(query)[0].page_content
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer in 1 sentence:"
    
    with st.chat_message("assistant"):
        response = llm(prompt)
        st.write(response)
