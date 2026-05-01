import streamlit as st
import os
import re
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="HR AI", layout="centered")
st.title("📄 HR Policy Bot (16GB RAM)")

# --- 1. DATA LOADING ---
@st.cache_resource
def init_retriever():
    file_path = "HR Policy"
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' not found! Please upload it to the Files tab.")
        return None
    
    loader = TextLoader(file_path)
    docs = loader.load()
    
    # Remove the noise (long strings of numbers)
    docs[0].page_content = re.sub(r'(\d+\s+){5,}', '', docs[0].page_content)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(splits, embeddings).as_retriever()

retriever = init_retriever()

# --- 2. CHAT LOGIC ---
if query := st.chat_input("Ask about leave policy:"):
    st.chat_message("user").write(query)
    
    # HF Spaces automatically maps secrets to environment variables
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        st.error("GROQ_API_KEY not found! Add it in Space Settings -> Secrets.")
    elif retriever:
        llm = ChatGroq(api_key=api_key, model_name="llama-3.3-70b-versatile")
        
        prompt = ChatPromptTemplate.from_template("Answer in 3 bullet points using context: {context}\nQuestion: {question}")
        chain = (
            {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
             "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        
        with st.chat_message("assistant"):
            st.write(chain.invoke(query))
