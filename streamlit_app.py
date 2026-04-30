import streamlit as st
import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

# =========================================
# 1. Load HR Policy file
# =========================================
DATA_PATH = "HR Policy.txt"

try:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("HR Policy file not found")

    loader = TextLoader(DATA_PATH, encoding="utf-8")
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
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = splitter.split_documents(documents)
docs = [d for d in docs if d.page_content.strip() != ""]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

# =========================================
# 3. LLM (Ollama)
# =========================================
llm = Ollama(
    model="llama3",   # make sure: ollama pull llama3
    temperature=0
)

# =========================================
# 4. Prompt
# =========================================
prompt = PromptTemplate(
    template="""
You are an HR assistant.

Answer the question based ONLY on the context below.
- Summarize clearly in 2–3 sentences.
- Do not repeat the context verbatim.
- If the answer is not in the context, reply exactly: "Not found in HR policy."

Context:
{context}

Question:
{question}

Final Answer:
""",
    input_variables=["context", "question"]
)

# =========================================
# 5. QA Chain
# =========================================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# =========================================
# 6. Guard function
# =========================================
def clean_answer(result):
    answer = result.strip()

    if not answer:
        return "Not found in HR policy."

    if answer.lower().startswith(("context", "you are an hr assistant")):
        return "Not found in HR policy."

    return answer

# =========================================
# 7. Streamlit UI
# =========================================
st.title("🤖 HR Chatbot")

user_input = st.text_input("Ask a question about HR policy:")

if user_input:
    # ✅ IMPORTANT FIX → use "query"
    res = qa_chain.invoke({"query": user_input})

    answer = clean_answer(res["result"])

    st.markdown("### 🧠 Answer")
    st.write(answer)

    st.markdown("### 📄 Sources")
    for d in res["source_documents"]:
        st.write("-", d.metadata.get("source", "HR Policy"))
