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

st.set_page_config(page_title="HR AI Assistant", layout="centered")
st.title("⚡ HR Policy AI (Groq Powered)")

# --- 1. DATA LOADING & CLEANING ---
@st.cache_resource
def init_retriever():
    file_path = "HR Policy"
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' not found in GitHub!")
        return None
    
    loader = TextLoader(file_path)
    docs = loader.load()
    
    # Noise Filter: Removes long lists of numbers (18, 19, 20...)
    docs[0].page_content = re.sub(r'(\d+\s+){5,}', '', docs[0].page_content)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    
    # This runs on Streamlit CPU (Low memory usage)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(splits, embeddings).as_retriever(search_kwargs={"k": 2})

# --- 2. SECURE LLM SETUP ---
@st.cache_resource
def load_llm():
    # Pulls the hidden key from Streamlit Secrets
    return ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )

retriever = init_retriever()
llm = load_llm()

# --- 3. RAG PROMPT ---
prompt = ChatPromptTemplate.from_template("""
You are a professional HR assistant. 
Answer the question using ONLY the context provided.
Provide a concise answer in exactly 3 bullet points.

Context: {context}
Question: {question}
Answer:""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input("Ask a policy question (e.g., How many leaves?)"):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)
    
    if retriever and llm:
        # Create the LCEL Pipe
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt 
            | llm 
            | StrOutputParser()
        )
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing HR Policy..."):
                try:
                    response = chain.invoke(query)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")
