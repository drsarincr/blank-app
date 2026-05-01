import streamlit as st
import os
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyMuPDFLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Page config
st.set_page_config(page_title="HR Policy Chatbot", page_icon="🤖", layout="wide")

# Title
st.title("🤖 HR Policy Chatbot")
st.markdown("Ask questions about HR policies and get instant answers!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("This chatbot uses Qwen2.5 (local model) and RAG to answer HR policy questions.")
    
    # Model selection
    model_options = {
        "Qwen2.5-0.5B (Fast, Low Memory)": "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen2.5-1.5B (Balanced)": "Qwen/Qwen2.5-1.5B-Instruct",
        "Flan-T5-Large (Good Quality)": "google/flan-t5-large",
        "Flan-T5-Base (Fastest)": "google/flan-t5-base"
    }
    
    selected_model = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=0
    )
    
    model_name = model_options[selected_model]

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to load local LLM
@st.cache_resource
def load_local_llm(_model_name):
    st.sidebar.info(f"📥 Loading {_model_name}...")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(_model_name, trust_remote_code=True)
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if "qwen" in _model_name.lower():
            # For Qwen models
            model = AutoModelForCausalLM.from_pretrained(
                _model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.1
            )
        else:
            # For T5 models
            model = AutoModelForCausalLM.from_pretrained(
                _model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.1
            )
        
        # Create LangChain LLM
        llm = HuggingFacePipeline(pipeline=pipe)
        
        st.sidebar.success(f"✅ Model loaded on {device}")
        return llm
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        st.error("Model loading failed. Using demo mode.")
        return None

# Function to initialize the RAG system
@st.cache_resource
def initialize_rag(_llm):
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
        docs = DirectoryLoader(
            DATA_PATH, 
            glob="**/*.txt", 
            loader_cls=TextLoader,
            loader_kwargs={'autodetect_encoding': True}
        ).load()
        documents.extend(add_meta(docs, "txt"))
        st.sidebar.info(f"📄 Loaded {len(docs)} TXT files")
    except Exception as e:
        pass
    
    # Load DOCX files
    try:
        docs = DirectoryLoader(DATA_PATH, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
        documents.extend(add_meta(docs, "docx"))
        st.sidebar.info(f"📄 Loaded {len(docs)} DOCX files")
    except Exception as e:
        pass
    
    # Load PDF files
    try:
        docs = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyMuPDFLoader).load()
        documents.extend(add_meta(docs, "pdf"))
        st.sidebar.info(f"📄 Loaded {len(docs)} PDF files")
    except Exception as e:
        pass
    
    if len(documents) == 0:
        st.warning("⚠️ No documents loaded, using demo data")
        documents = [Document(
            page_content="""
            HR POLICY DOCUMENT
            
            Leave Policy:
            - Employees are entitled to 20 days of paid leave per year
            - Sick leave: 10 days per year
            - Casual leave: 10 days per year
            
            Notice Period:
            - All employees must provide 30 days notice before resignation
            - Management reserves the right to waive notice period
            
            Benefits:
            - Health insurance provided to all full-time employees
            - Provident fund contribution: 12% of basic salary
            - Annual bonus based on performance
            
            Working Hours:
            - Standard working hours: 9 AM to 6 PM
            - Lunch break: 1 hour
            - Flexible working hours available on approval
            """,
            metadata={"source": "demo_policy"}
        )]
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    docs = [d for d in docs if d.page_content and d.page_content.strip() != ""]
    
    st.sidebar.success(f"✅ Created {len(docs)} document chunks")
    
    # Create embeddings and vector store
    with st.spinner("🔄 Creating embeddings..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create prompt
    prompt = PromptTemplate(
        template="""You are an HR assistant. Answer the question based only on the context below.
If the answer is not in the context, say "Not found in HR policy."

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    if _llm:
        qa_chain = RetrievalQA.from_chain_type(
            llm=_llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        return qa_chain
    else:
        return None

# Load LLM
if st.session_state.qa_chain is None:
    with st.spinner("🔄 Loading model and initializing RAG system..."):
        llm = load_local_llm(model_name)
        if llm:
            st.session_state.qa_chain = initialize_rag(llm)
        else:
            st.error("Failed to initialize the system. Please check the logs.")
            st.stop()

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
                    st.error(f"Error generating response: {str(e)}")
