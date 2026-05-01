import streamlit as st
import os

# Modern LangChain imports (Avoids legacy chains)
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="HR AI Assistant", layout="centered")
st.title("📄 HR Policy Chatbot")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    st.info("Since this is hosted on Streamlit Cloud, you must provide a public URL to your Ollama instance (e.g., via Ngrok).")
    ollama_url = st.text_input("Ollama Endpoint URL", value="http://localhost:11434")

# --- 1. DATA LOADING LOGIC ---
@st.cache_resource
def init_retriever():
    # Look for the path in the current directory
    data_path = os.path.join(os.getcwd(), "HR Policy")
    
    if not os.path.exists(data_path):
        st.error(f"❌ Path not found: {data_path}. Please check your GitHub repo.")
        return None

    try:
        # Detect if 'HR Policy' is a folder or a file
        if os.path.isdir(data_path):
            st.toast("Loading from folder: HR Policy")
            loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader)
        else:
            st.toast("Loading from single file: HR Policy")
            loader = TextLoader(data_path)
            
        documents = loader.load()
        
        if not documents:
            st.warning("No content found in the selected path.")
            return None

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        
        # Initialize Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Build Vector Store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
        
    except Exception as e:
        st.error(f"Processing Error: {e}")
        return None

retriever = init_retriever()

# --- 2. RAG CHAIN LOGIC (LCEL) ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """
You are an HR assistant. Answer the question using ONLY the context provided below. 
If the information is not in the context, strictly say: "Not found in HR policy."

Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_query := st.chat_input("Ask a policy question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if retriever:
        try:
            # Initialize Ollama
            llm = OllamaLLM(model="llama3.2", base_url=ollama_url)
            
            # Create the LCEL Pipe
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            with st.chat_message("assistant"):
                with st.spinner("Searching policies..."):
                    response = chain.invoke(user_query)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
        except Exception as e:
            st.error(f"Connection Error: Could not connect to Ollama at {ollama_url}. {e}")
    else:
        st.warning("Retriever not initialized. Please fix the data path issues.")
