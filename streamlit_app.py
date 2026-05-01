import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="HR Policy Bot", layout="centered")
st.title("📄 HR Policy Chatbot")

# --- OLLAMA CONFIGURATION ---
# Note: Streamlit Cloud cannot run Ollama locally. 
# You must provide the URL to your hosted Ollama instance (e.g., via Ngrok)
OLLAMA_URL = st.sidebar.text_input("Ollama Endpoint URL", value="http://localhost:11434")

# --- 1. INITIALIZE COMPONENTS ---
@st.cache_resource
def load_data():
    DATA_PATH = "HR Policy" # Ensure this folder exists in your GitHub repo
    
    if not os.path.exists(DATA_PATH):
        st.error(f"Folder '{DATA_PATH}' not found in repository!")
        return None

    # Load only .txt files
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    
    # Create Embeddings and Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = load_data()

# --- 2. PROMPT SETUP ---
template = """
You are an HR assistant. Answer ONLY from the context provided.
If the answer is not in the context, say: "Not found in HR policy."

Context: {context}
Question: {question}
Answer:"""

prompt_template = PromptTemplate(
    template=template, 
    input_variables=["context", "question"]
)

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a question about HR policies..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if retriever:
        try:
            llm = OllamaLLM(model="llama3.2", base_url=OLLAMA_URL)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt_template}
            )
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    res = qa_chain.invoke({"query": user_input})
                    answer = res["result"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Connection Error: Is Ollama running at {OLLAMA_URL}?")
