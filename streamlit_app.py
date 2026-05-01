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

st.set_page_config(page_title="HR AI Assistant", layout="wide")
st.title("📄 HR Policy Chatbot")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("1. Model Selection")
    model_choice = st.selectbox("LLM Model", ["qwen2.5", "llama3.2", "mistral"])
    
    st.header("2. Connection Settings")
    st.markdown("If using Streamlit Cloud, use an **Ngrok URL** below.")
    endpoint = st.text_input("Ollama URL", value="http://localhost:11434")
    
    st.divider()
    st.info("The app indexes policy files on the cloud server, but sends questions to your Ollama URL.")

# --- 3. VECTOR DB INITIALIZATION ---
@st.cache_resource
def setup_retriever():
    # Path inside the GitHub repo
    data_path = os.path.join(os.getcwd(), "HR Policy")
    
    if not os.path.exists(data_path):
        return f"Error: '{data_path}' not found in GitHub repository."

    try:
        # Step 1: Load Data (Folder vs File)
        if os.path.isdir(data_path):
            loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader)
        else:
            loader = TextLoader(data_path)
        
        docs = loader.load()
        if not docs:
            return "Error: No text found in 'HR Policy'."

        # Step 2: Split and Embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        
        # This part runs on the Streamlit Cloud CPU
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        return f"Error during indexing: {str(e)}"

# Run the initialization
retriever = setup_retriever()

# --- 4. RAG LOGIC ---
if isinstance(retriever, str):
    st.error(retriever)
else:
    st.success("✅ HR Policies indexed and ready.")

    # Define the Chain components
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template("""
    You are an HR assistant. Answer ONLY based on the context.
    If the answer isn't there, say "Not found in policy."
    
    Context: {context}
    Question: {question}
    Answer:
    """)

    # --- 5. CHAT UI ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_query := st.chat_input("Ask a policy question..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        try:
            # Initialize connection to your local Ollama (via Endpoint)
            llm = OllamaLLM(model=model_choice, base_url=endpoint)
            
            # Modern LCEL Chain
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            with st.chat_message("assistant"):
                with st.spinner("Consulting Policies..."):
                    response = chain.invoke(user_query)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        
        except Exception as e:
            st.error(f"❌ Connection Failed to {endpoint}")
            st.markdown(f"""
            **How to fix this:**
            1. Ensure Ollama is running on your PC.
            2. If you are on the Streamlit Cloud website, you cannot use `localhost`. 
            3. Run `ngrok http 11434` on your PC and paste the **https** link into the sidebar.
            """)
