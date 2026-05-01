import streamlit as st
import os
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyMuPDFLoader,
    Docx2txtLoader, UnstructuredPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document

# Page config
st.set_page_config(page_title="HR Policy Chatbot", page_icon="🤖", layout="wide")

# Title
st.title("🤖 HR Policy Chatbot")
st.markdown("Ask questions about HR policies and get instant answers!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("This chatbot uses Groq (Llama 3.1) and RAG to answer HR policy questions.")
    
    # API Key input
    groq_api_key = st.text_input("Groq API Key", type="password", value=st.secrets.get("GROQ_API_KEY", ""))
    
    if not groq_api_key:
        st.warning("⚠️ Please enter your Groq API Key")
        st.markdown("[Get free API key here](https://console.groq.com/)")
        st.stop()

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to initialize the RAG system
@st.cache_resource
def initialize_rag(_api_key):
    DATA_PATH = "HR Policy"
    
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Path not found: {DATA_PATH}")
        return None
    
    # Load documents
    documents = []
    
    def add_meta(docs, tag):
        for d in docs:
            d.metadata["source"] = d.metadata.get("source", tag)
        return docs
    
    # Load TXT files
    try:
        docs = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader).load()
        documents.extend(add_meta(docs, "txt"))
    except:
        pass
    
    # Load DOCX files
    try:
        docs = DirectoryLoader(DATA_PATH, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
        documents.extend(add_meta(docs, "docx"))
    except:
        pass
    
    # Load PDF files (fast)
    try:
        docs = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyMuPDFLoader).load()
        documents.extend(add_meta(docs, "pdf_fast"))
    except:
        pass
    
    if len(documents) == 0:
        st.warning("⚠️ No documents loaded, using demo data")
        documents = [Document(
            page_content="Employees get 20 days leave. Notice period is 30 days.",
            metadata={"source": "demo"}
        )]
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    docs = [d for d in docs if d.page_content and d.page_content.strip() != ""]
    
    st.sidebar.success(f"✅ Loaded {len(docs)} document chunks")
    
    # Create embeddings and vector store
    with st.spinner("🔄 Creating embeddings..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Initialize LLM with Groq
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=_api_key,
        temperature=0
    )
    
    # Create prompt
    prompt = PromptTemplate(
        template="""You are an HR assistant.
Answer ONLY from the context provided below.
If the answer is not found in the context, say: "Not found in HR policy."

Context:
{context}

Question:
{question}

Answer:""",
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain

# Initialize RAG if not done
if st.session_state.qa_chain is None:
    with st.spinner("🔄 Initializing RAG system..."):
        st.session_state.qa_chain = initialize_rag(groq_api_key)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📄 View Sources"):
                for source in message["sources"]:
                    st.text(f"- {source}")

# Chat input
if prompt := st.chat_input("Ask about HR policies..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    if st.session_state.qa_chain:
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = st.session_state.qa_chain.invoke({"query": prompt})
                    answer = response["result"]
                    sources = [d.metadata.get("source", "Unknown") for d in response["source_documents"]]
                    
                    st.markdown(answer)
                    with st.expander("📄 View Sources"):
                        for source in sources:
                            st.text(f"- {source}")
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
