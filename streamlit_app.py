import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# =========================================
# 1. Load HR Policy file
# =========================================
DATA_PATH = "HR Policy"   # adjust if file is HR Policy.txt
try:
    loader = TextLoader(DATA_PATH)
    documents = loader.load()
except Exception:
    from langchain_core.documents import Document
    documents = [Document(
        page_content="Employees get 20 days leave. Notice period is 30 days.",
        metadata={"source": "demo"}
    )]

# =========================================
# 2. Split + Embed
# =========================================
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)
docs = [d for d in docs if d.page_content.strip() != ""]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =========================================
# 3. LLM (Hugging Face, CPU-friendly)
# =========================================
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")
llm = HuggingFacePipeline(pipeline=generator)

prompt = PromptTemplate(
    template="""
You are an HR assistant.

Use the context below to answer the question concisely.
- Summarize clearly in 2–3 sentences.
- Do not repeat the context verbatim.
- If the answer is not in the context, reply: "Not found in HR policy."

Context:
{context}

Question:
{question}

Final Answer:
""",
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# =========================================
# 4. Streamlit UI
# =========================================
st.title("🤖 HR Chatbot")

user_input = st.text_input("Ask a question about HR policy:")

if user_input:
    res = qa_chain.invoke({"query": user_input})
    answer = res["result"].strip()

    # Guard: if answer looks like raw context, replace
    if "Context:" in answer or len(answer.split()) > 120:
        answer = "Not found in HR policy."

    st.markdown("### 🧠 Answer")
    st.write(answer)

    st.markdown("### 📄 Sources")
    for d in res["source_documents"]:
        st.write("-", d.metadata.get("source"))
