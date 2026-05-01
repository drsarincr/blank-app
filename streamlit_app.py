import streamlit as st
import os
import subprocess

# Standard LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="HR Policy Chatbot", layout="wide")

st.title("🤖 HR Policy AI Assistant")
st.markdown("Query your `.txt` policy files using RAG.")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("Settings")
    # Streamlit Cloud cannot run Ollama locally. 
    # Use an Ngrok URL or a local tunnel URL here.
    ollama_url = st.text_input("Ollama Endpoint", value="http://localhost:11434")
    st.info("Note: If running on Streamlit Cloud, 'localhost' won't work. Use a tunnel URL (like Ngrok).")

# --- 1. DATA LOADING & VECTOR DB ---
@st.cache_resource
def initialize_rag():
    DATA_PATH = "HR Policy"
    
    if not os.path.exists(DATA_PATH):
        st.error(f"Directory '{DATA_PATH}' not found in GitHub repo.")
        return None

    # Load TXT files
    try:
        loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        if not documents:
            st.warning("No .txt files found in the HR Policy folder.")
            return None

        # Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        all_splits = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Vector Store
        vectorstore = FAISS.from_documents(all_splits, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.error(f"Error initializing RAG: {e}")
        return None

retriever = initialize_rag()

# --- 2. LLM & PROMPT SETUP ---
prompt_template = """
You are a professional HR assistant. Use the following context to answer the question.
If the answer is not in the context, strictly say: "Not found in HR policy."
Do not make up answers.

Context: {context}
Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_query := st.chat_input("Ask a policy question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if retriever is None:
        st.error("RAG system not initialized. Check your 'HR Policy' folder.")
    else:
        try:
            # Initialize Ollama via the provided URL
            llm = OllamaLLM(model="llama3.2", base_url=ollama_url)
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": QA_PROMPT},
                return_source_documents=True
            )

            with st.chat_message("assistant"):
                with st.spinner("Analyzing policy documents..."):
                    response = qa_chain.invoke({"query": user_query})
                    answer = response["result"]
                    
                    st.markdown(answer)
                    
                    # Optional: Show sources
                    with st.expander("View Sources"):
                        for doc in response["source_documents"]:
                            st.write(f"- {doc.metadata.get('source', 'Unknown source')}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Connection Error: Could not connect to Ollama at {ollama_url}. Error: {e}")
