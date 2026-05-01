import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Cloud AI HR Assistant", layout="centered")
st.title("📄 HR Policy Bot (100% Cloud)")
st.info("Note: This runs on a free CPU. The first question might take a minute to load the model.")

# --- 1. VECTOR DB SETUP ---
@st.cache_resource
def init_retriever():
    data_path = os.path.join(os.getcwd(), "HR Policy")
    
    if not os.path.exists(data_path):
        # Create dummy data if folder is missing so it doesn't crash
        st.error("HR Policy path not found. Please check GitHub folder naming.")
        return None

    try:
        if os.path.isdir(data_path):
            loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader)
        else:
            loader = TextLoader(data_path)
            
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 2})
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- 2. CLOUD LLM SETUP (NO PC / NO API KEY) ---
@st.cache_resource
def load_cloud_llm():
    # We use a tiny model that fits in Streamlit's limited memory
    llm = CTransformers(
        model="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        model_type="llama",
        config={'max_new_tokens': 256, 'temperature': 0.1}
    )
    return llm

retriever = init_retriever()
llm = load_cloud_llm()

# --- 3. RAG LOGIC ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template("""
Context: {context}
Question: {question}
Answer only using the context. If not found, say 'Not in policy'.
Answer:""")

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input("Ask about HR Policy..."):
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
            with st.spinner("Thinking (CPU Mode)..."):
                response = chain.invoke(query)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
