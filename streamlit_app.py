import streamlit as st
import os

# Essential LCEL imports
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
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")

# --- 1. SETUP RETRIEVER ---
@st.cache_resource
def get_retriever():
    # Use absolute path to avoid ambiguity
    base_path = os.path.dirname(__file__)
    data_dir = os.path.join(base_path, "HR Policy")
    
    # Check if the path exists AND is a directory
    if not os.path.exists(data_dir):
        st.error(f"❌ Path not found: {data_dir}. Please ensure a FOLDER named 'HR Policy' exists in your GitHub repo.")
        return None
        
    if not os.path.isdir(data_dir):
        st.error(f"❌ '{data_dir}' is a file, but we need a FOLDER. Please rename or move your file into a folder.")
        return None
    
    try:
        # Load .txt files from the directory
        loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader)
        docs = loader.load()
        
        if not docs:
            st.warning("⚠️ Folder found, but no .txt files were inside.")
            return None
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return None

retriever = get_retriever()

# --- 2. RAG LOGIC ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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
            llm = OllamaLLM(model="llama3.2", base_url=ollama_url)
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )
            with st.chat_message("assistant"):
                response = chain.invoke(user_query)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Ollama Connection Error. Is it running at {ollama_url}?")
