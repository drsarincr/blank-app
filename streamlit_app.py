import streamlit as st
import os

# Modern LangChain imports
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Local AI HR Assistant", layout="centered")
st.title("📄 HR Policy Chatbot (No API Key)")

# --- 1. CONFIGURATION ---
with st.sidebar:
    st.header("Model Settings")
    # You can change 'llama3.2' to 'qwen2.5' or 'mistral' here
    model_name = st.selectbox("Select Model", ["llama3.2", "qwen2.5", "mistral"], index=0)
    
    st.header("Connection")
    st.markdown("""
    **If hosting on Streamlit Cloud:** 1. Run `ngrok http 11434` on your PC.  
    2. Paste the **https** URL below.
    """)
    endpoint_url = st.text_input("Ollama Endpoint", value="http://localhost:11434")

# --- 2. DATA LOADING (CACHED) ---
@st.cache_resource
def load_hr_data():
    # Path to your HR Policy folder/file
    data_path = os.path.join(os.getcwd(), "HR Policy")
    
    if not os.path.exists(data_path):
        return "ERROR: 'HR Policy' folder not found in repository."

    try:
        # Check if it's a folder or a single file
        if os.path.isdir(data_path):
            loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader)
        else:
            loader = TextLoader(data_path)
            
        docs = loader.load()
        if not docs:
            return "ERROR: No text found in documents."

        # Split text into manageable chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        # Create Embeddings (This runs on the Streamlit Server CPU)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Build Vector Database
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
        
    except Exception as e:
        return f"ERROR: {str(e)}"

# Initialize the vector store
result = load_hr_data()

if isinstance(result, str) and result.startswith("ERROR"):
    st.error(result)
    retriever = None
else:
    retriever = result.as_retriever(search_kwargs={"k": 3})
    st.sidebar.success("✅ HR Policies Indexed")

# --- 3. RAG CHAIN SETUP ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful HR Assistant. Use the provided context to answer the question.
If the answer is not in the context, say: "I'm sorry, that is not covered in our HR policy."

Context:
{context}

Question: {question}
Answer:""")

# --- 4. CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_input := st.chat_input("Ask a question about HR policy..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if retriever:
        try:
            # Initialize Ollama Connection
            llm = OllamaLLM(model=model_name, base_url=endpoint_url)
            
            # The LCEL Pipe (The Modern Chain)
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            with st.chat_message("assistant"):
                with st.spinner("Searching policies..."):
                    response = chain.invoke(user_input)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
        except Exception as e:
            st.error(f"❌ Connection Failed: Ensure Ollama is running and accessible at {endpoint_url}")
            st.info("If you are on Streamlit Cloud, remember you need a tunnel like Ngrok.")
