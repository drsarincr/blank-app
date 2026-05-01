import streamlit as st
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="HR Policy AI", layout="centered")
st.title("📄 HR Policy Bot")
st.caption("Using Gemma-2B (Optimized for Cloud CPU)")

# --- 1. DATA LOADING ---
@st.cache_resource
def init_retriever():
    data_path = os.path.join(os.getcwd(), "HR Policy")
    if not os.path.exists(data_path):
        return None

    try:
        loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader) if os.path.isdir(data_path) else TextLoader(data_path)
        docs = loader.load()
        # Very small chunks to keep the CPU fast
        splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
        splits = splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 1}) # Only take 1 best match to save memory
    except Exception:
        return None

# --- 2. LIGHTWEIGHT MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        # Gemma 2B is much lighter and stable for Streamlit Cloud
        model_url = "https://huggingface.co/lmstudio-community/gemma-2b-it-GGUF/resolve/main/gemma-2b-it-Q4_K_M.gguf"
        
        return LlamaCpp(
            model_url=model_url,
            temperature=0.1,
            max_tokens=128,
            top_p=1,
            n_ctx=1024, 
            verbose=False,
        )
    except Exception as e:
        st.error(f"Cloud CPU Memory Limit Reached. Try refreshing. Error: {e}")
        return None

retriever = init_retriever()
llm = load_model()

# --- 3. RAG CHAIN ---
def format_docs(docs):
    # Only return the first few words to prevent 'number noise'
    content = docs[0].page_content if docs else ""
    return content[:300] 

prompt = ChatPromptTemplate.from_template("""
<start_of_turn>user
Answer the question in exactly 3 lines using only this context:
Context: {context}
Question: {question}<end_of_turn>
<start_of_turn>model
""")

# --- 4. UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input("Ask about leave..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    if retriever and llm:
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        with st.chat_message("assistant"):
            with st.spinner("Processing (Cloud CPU)..."):
                try:
                    response = chain.invoke(query)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error("The server is too busy. Please try again in a moment.")
