import streamlit as st
import os

# Essential LCEL imports (No legacy chains)
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="HR Policy Assistant", layout="centered")
st.title("📄 HR Policy Bot")

# --- SETTINGS ---
with st.sidebar:
    st.header("Connection")
    # Rememeber: 'localhost' won't work on Streamlit Cloud. Use your Ngrok URL.
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")

# --- 1. SETUP RETRIEVER ---
@st.cache_resource
def get_retriever():
    DATA_PATH = "HR Policy"
    if not os.path.exists(DATA_PATH):
        st.error(f"Folder '{DATA_PATH}' not found in GitHub!")
        return None
    
    # Load .txt files
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    
    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    
    # Embeddings (Runs on Streamlit CPU)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Vector Store
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = get_retriever()

# --- 2. RAG LOGIC ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Simple template
prompt = ChatPromptTemplate.from_template("""
You are an HR assistant. Answer based ONLY on the context below. 
If not in context, say "Policy not found."

Context:
{context}

Question: {question}
Answer:
""")

# --- 3. CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

if user_query := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    if retriever:
        try:
            # Initialize Ollama
            llm = OllamaLLM(model="llama3.2", base_url=ollama_url)
            
            # THE LCEL PIPE: This replaces all 'Chain' imports
            # 1. Retrieve -> 2. Format -> 3. Prompt -> 4. LLM -> 5. Parse
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            with st.chat_message("assistant"):
                response = chain.invoke(user_query)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Connection failed. Is Ollama running at {ollama_url}?")
            st.debug(e)
