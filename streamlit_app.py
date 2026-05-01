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

st.set_page_config(page_title="Advanced HR AI", layout="centered")
st.title("📄 High-Accuracy HR Bot")
st.caption("Running Phi-3 Mini (3.8B) directly on Streamlit Cloud CPU")

# --- 1. VECTOR DATABASE ---
@st.cache_resource
def init_retriever():
    data_path = os.path.join(os.getcwd(), "HR Policy")
    if not os.path.exists(data_path):
        st.error("Missing 'HR Policy' folder/file.")
        return None

    loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader) if os.path.isdir(data_path) else TextLoader(data_path)
    docs = loader.load()
    
    # Smaller chunks help the model focus better on the specific '15 days' rule
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    splits = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

# --- 2. PHI-3 MODEL SETUP (BETTER BRAIN) ---
@st.cache_resource
def load_better_llm():
    # Phi-3 is much more capable of summarizing and following logic than TinyLlama
    return CTransformers(
        model="TheBloke/Phi-3-mini-4k-instruct-GGUF",
        model_file="phi-3-mini-4k-instruct.Q4_K_M.gguf",
        model_type="phi3",
        config={'max_new_tokens': 512, 'temperature': 0.1, 'context_length': 2048}
    )

retriever = init_retriever()
llm = load_better_llm()

# --- 3. RAG CHAIN ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Instructions added to help the model ignore the 'noise' (long lists of numbers)
prompt = ChatPromptTemplate.from_template("""
<|system|>
You are a precise HR Assistant. Answer ONLY using the context provided. 
If the answer is a number, double-check it. 
Ignore repetitive lists of numbers if they don't answer the question.
<|end|>
<|user|>
Context: {context}
Question: {question}
<|end|>
<|assistant|>
""")

# --- 4. UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input("Ask about leave policy..."):
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
            with st.spinner("Processing with Phi-3 (CPU)..."):
                response = chain.invoke(query)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
