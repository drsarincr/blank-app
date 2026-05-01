import streamlit as st
import os
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyMuPDFLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# Page config
st.set_page_config(page_title="HR Policy Chatbot", page_icon="🤖", layout="wide")

# Title
st.title("🤖 HR Policy Chatbot")
st.markdown("Ask questions about HR policies and get instant answers!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("This chatbot uses Flan-T5 (local model) and RAG to answer HR policy questions.")

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to load local LLM
@st.cache_resource
def load_local_llm():
    st.sidebar.info("📥 Loading Flan-T5 model...")
    
    try:
        # Using Flan-T5-Base (lightweight, works on Streamlit Cloud free tier)
        model_name = "google/flan-t5-base"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Create pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            do_sample=False,
            temperature=0.1
        )
        
        # Create LangChain LLM
        llm = HuggingFacePipeline(pipeline=pipe)
        
        st.sidebar.success("✅ Flan-T5 model loaded successfully")
        return llm
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None

# Function to initialize the RAG system
@st.cache_resource
def initialize_rag(_llm):
    DATA_PATH = "HR Policy"
    
    # Load documents
    documents = []
    
    def add_meta(docs, tag):
        for d in docs:
            d.metadata["source"] = d.metadata.get("source", tag)
        return docs
    
    # Check if directory exists
    if os.path.exists(DATA_PATH):
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
            st.sidebar.warning(f"TXT: {str(e)[:50]}")
        
        # Load DOCX files
        try:
            docs = DirectoryLoader(DATA_PATH, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
            documents.extend(add_meta(docs, "docx"))
            st.sidebar.info(f"📄 Loaded {len(docs)} DOCX files")
        except Exception as e:
            st.sidebar.warning(f"DOCX: {str(e)[:50]}")
        
        # Load PDF files
        try:
            docs = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyMuPDFLoader).load()
            documents.extend(add_meta(docs, "pdf"))
            st.sidebar.info(f"📄 Loaded {len(docs)} PDF files")
        except Exception as e:
            st.sidebar.warning(f"PDF: {str(e)[:50]}")
    
    # Use demo data if no documents loaded
    if len(documents) == 0:
        st.warning("⚠️ No documents found in 'HR Policy' folder. Using demo data.")
        documents = [Document(
            page_content="""
HR POLICY DOCUMENT

LEAVE POLICY:
- Annual Leave: Employees are entitled to 20 days of paid leave per year
- Sick Leave: 10 days per year with medical certificate
- Casual Leave: 10 days per year
- Maternity Leave: 180 days for female employees
- Paternity Leave: 15 days for male employees

NOTICE PERIOD:
- All employees must provide 30 days written notice before resignation
- During probation period: 15 days notice required
- Management can waive notice period at their discretion

EMPLOYEE BENEFITS:
- Health Insurance: Comprehensive coverage for employee and family
- Provident Fund: Company contributes 12% of basic salary
- Annual Bonus: Performance-based bonus up to 3 months salary
- Professional Development: Up to $2000 per year for courses

WORKING HOURS:
- Standard Hours: Monday to Friday, 9:00 AM to 6:00 PM
- Lunch Break: 1 hour (1:00 PM to 2:00 PM)
- Flexible Hours: Available with manager approval
- Remote Work: Hybrid model - 3 days office, 2 days remote

COMPENSATION:
- Salary Review: Annual review in January
- Increment: Based on performance (3-15%)
- Overtime Pay: 1.5x hourly rate for approved overtime
            """,
            metadata={"source": "demo_hr_policy"}
        )]
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = splitter.split_documents(documents)
    docs = [d for d in docs if d.page_content and d.page_content.strip() != ""]
    
    st.sidebar.success(f"✅ Created {len(docs)} document chunks")
    
    # Create embeddings and vector store
    with st.spinner("🔄 Creating embeddings..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create prompt
    prompt = PromptTemplate(
        template="""Answer the question based only on the following context. If the answer is not in the context, say "Not found in HR policy."

Context: {context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=_llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain

# Initialize system
if st.session_state.qa_chain is None:
    with st.spinner("🔄 Loading model and initializing RAG system..."):
        llm = load_local_llm()
        if llm:
            st.session_state.qa_chain = initialize_rag(llm)
        else:
            st.error("Failed to load the model. Please refresh the page.")
            st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📄 View Sources"):
                for source in message["sources"]:
                    st.text(f"• {source}")

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
                    sources = list(set([d.metadata.get("source", "Unknown") for d in response["source_documents"]]))
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📄 View Sources"):
                            for source in sources:
                                st.text(f"• {source}")
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Please try rephrasing
