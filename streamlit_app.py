import streamlit as st
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Cloud HR AI", layout="centered")
st.title("📄 High-Accuracy HR Bot")
st.caption("Running Phi-3 (3.8B) - 100% Cloud CPU")

# --- 1. DATA LOADER ---
@st.cache_resource
def init_retriever():
    data_path = os.path.join(os.getcwd(), "HR Policy")
    if not os.path.exists(data_path):
        st.error("Missing 'HR Policy' folder/file in GitHub.")
        return None

    try:
        if os.path.isdir(data_path):
            loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader)
        else:
            loader = TextLoader(data_path)
            
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        splits = splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 2})
    except Exception as e:
        st.error(f"Retriever error: {e}")
        return None

# --- 2. UPDATED MODEL SOURCE ---
@st.cache_resource
def load_llm():
    try:
        # Switched to a reliable current repository for Phi-3 GGUF
        return CTransformers(
            model="bartowski/Phi-3-mini-4k-instruct-GGUF",
            model_file="Phi-3-mini-4k-instruct-Q4_K_M.gguf",
            model_type="phi3",
            config={'max_new_tokens': 512, 'temperature': 0.1, 'context_length': 2048}
        )
    except Exception as e:
        st.error(f"Model download failed: {e}. Please refresh the page.")
        return None

retriever = init_retriever()
llm = load_llm()

# --- 3. RAG PIPELINE ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template("""
<|system|>
You are a helpful HR Assistant. Answer using the context. 
Be concise and strictly summarize in 3 lines.
<|end|>
<|user|>
Context: {context}
Question: {question}
<|end|>
<|assistant|>
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input("Ask a question..."):
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
            with st.spinner("Analyzing (this may take 45+ seconds on CPU)..."):
                try:
                    response = chain.invoke(query)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Generation error: {e}")
