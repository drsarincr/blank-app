import streamlit as st
import os
import re

# NEW MODERN IMPORTS
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- DATA LOADING (CACHED) ---
@st.cache_resource
def init_retriever():
    file_path = "HR Policy"
    if not os.path.exists(file_path):
        return None
    
    loader = TextLoader(file_path)
    docs = loader.load()
    
    # Clean the numbers noise (18, 19, 20...)
    docs[0].page_content = re.sub(r'(\d+\s+){5,}', '', docs[0].page_content)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(splits, embeddings).as_retriever()

# --- THE NEW CHAIN LOGIC ---
# This replaces the old RetrievalQA
def get_answer(query, retriever):
    llm = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"], 
        model_name="llama-3.3-70b-versatile"
    )
    
    prompt = ChatPromptTemplate.from_template("""
    Use context to answer in 3 lines:
    Context: {context}
    Question: {question}
    """)
    
    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain.invoke(query)
