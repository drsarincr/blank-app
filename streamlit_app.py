import streamlit as st
import os
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="HR Policy AI", layout="centered")
st.title("📄 Modern HR Policy Bot")

# --- SETTINGS ---
with st.sidebar:
    ollama_url = st.text_input("Ollama Endpoint", value="http://localhost:11434")
    st.caption("On Streamlit Cloud, replace 'localhost' with your Ngrok/Tunnel URL.")

# --- 1. VECTOR STORE SETUP ---
@st.cache_resource
def get_retriever():
    DATA_PATH = "HR Policy"
    if not os.path.exists(DATA_PATH):
        st.error("Data folder not found!")
        return None
    
    # Text Loader Only
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    final_docs = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(final_docs, embeddings)
    return vectorstore.as_retriever()

retriever = get_retriever()

# --- 2. THE MODERN CHAIN ---
def get_qa_chain(user_url):
    llm = OllamaLLM(model="llama3.2", base_url=user_url)
    
    # Modern Chat Prompt
    system_prompt = (
        "You are an HR assistant. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer, say that you "
        "don't know. Answer ONLY from the context.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create the modern chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# --- 3. UI LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)
    
    if retriever:
        try:
            rag_chain = get_qa_chain(ollama_url)
            with st.chat_message("assistant"):
                with st.spinner("Searching policies..."):
                    response = rag_chain.invoke({"input": query})
                    answer = response["answer"]
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Error: {e}")
