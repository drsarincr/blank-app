import streamlit as st
import os

from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyMuPDFLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub


# =========================================
# 1. UI
# =========================================
st.set_page_config(page_title="HR Chatbot", layout="centered")
st.title("🤖 HR Chatbot (Deployable Version)")


# =========================================
# 2. DATA PATH
# =========================================
DATA_PATH = "HR Policy"

if not os.path.exists(DATA_PATH):
    st.error("❌ 'HR Policy' folder not found")
    st.stop()


# =========================================
# 3. LOAD DATA
# =========================================
@st.cache_resource
def load_data(path):

    documents = []

    def add_meta(docs, tag):
        for d in docs:
            d.metadata["source"] = d.metadata.get("source", tag)
        return docs

    try:
        docs = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader).load()
        documents.extend(add_meta(docs, "txt"))
    except:
        pass

    try:
        docs = DirectoryLoader(path, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
        documents.extend(add_meta(docs, "docx"))
    except:
        pass

    try:
        docs = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyMuPDFLoader).load()
        documents.extend(add_meta(docs, "pdf"))
    except:
        pass

    # fallback
    if len(documents) == 0:
        from langchain_core.documents import Document
        documents = [Document(
            page_content="Employees get 20 days leave. Notice period is 30 days.",
            metadata={"source": "demo"}
        )]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embeddings)


vectorstore = load_data(DATA_PATH)


# =========================================
# 4. LLM (CLOUD)
# =========================================
token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

if not token:
    st.error("❌ Add HuggingFace token in Streamlit Secrets")
    st.stop()

llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    huggingfacehub_api_token=token,
    model_kwargs={"temperature": 0}
)


# =========================================
# 5. PROMPT
# =========================================
prompt = PromptTemplate(
    template="""
You are an HR assistant.

Answer ONLY from the context.
If not found, say: "Not found in HR policy."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


# =========================================
# 6. CHAIN
# =========================================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)


# =========================================
# 7. CHAT
# =========================================
query = st.text_input("Ask your HR question:")

if query:
    res = qa_chain.invoke({"query": query})

    st.markdown("### 🧠 Answer")
    st.write(res["result"])

    st.markdown("### 📄 Sources")
    for d in res["source_documents"]:
        st.write("-", d.metadata.get("source"))
