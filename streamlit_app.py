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

st.set_page_config(page_title="HR AI Assistant", layout="centered")
st.title("📄 HR Policy Chatbot")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    ollama_url = st.text_input("Ollama Endpoint URL", value="http://localhost:11434")
    st.caption("Note: Localhost only works for local dev. Use Ngrok for Cloud.")

# --- 1. DATA LOADING LOGIC ---
# We keep ONLY logic here. No st.toast or st.error inside the cache.
@st.cache_resource
def load_vector_store():
    data_path = os.path.join(os.getcwd(), "HR Policy")
    
    if not os.path.exists(data_path):
        return "ERROR: Path not found"

    try:
        if os.path.isdir(data_path):
            loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader)
        else:
            loader = TextLoader(data_path)
            
        documents = loader.load()
        if not documents:
            return "ERROR: No content"

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
        
    except Exception as e:
        return f"ERROR: {str(e)}"

# --- 2. INITIALIZE RETRIEVER (Outside Cache) ---
result = load_vector_store()

if isinstance(result, str) and result.startswith("ERROR"):
    st.error(result)
    retriever = None
else:
    st.success("HR Policies Loaded Successfully!")
    retriever = result.as_retriever(search_kwargs={"k": 3})

# --- 3. RAG CHAIN LOGIC ---
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

# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a policy question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if retriever:
        try:
            llm = OllamaLLM(model="llama3.2", base_url=ollama_url)
            
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = chain.invoke(user_query)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
        except Exception as e:
            st.error(f"Connection Error: Ensure Ollama is reachable. {e}")
