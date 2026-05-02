# =========================================
# STREAMLIT HR CHATBOT APP
# =========================================
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# =========================================
# 1. HR POLICY (SELF-CONTAINED)
# =========================================
hr_policy = """
The company provides a structured leave policy to ensure employee well-being while maintaining operational efficiency.
Employees are entitled to 20 days of paid leave annually, including casual, sick, and earned leave.
Casual leave is limited to 8 days per year for personal matters. Sick leave includes 7 days annually and requires
medical proof if exceeding two consecutive days. Earned leave accumulates over time and can be carried forward up to 30 days.
Employees must apply for leave at least 3 days in advance for planned absences. Emergency leave must be reported immediately.
Unapproved leave may lead to salary deduction or disciplinary action.
Maternity leave is 26 weeks as per law. Paternity leave is 5 days. Adoption leave is also supported.
Public holidays are separate from leave balance. Employees working on holidays receive compensatory off.
Earned leave can be encashed during resignation. Casual and sick leave cannot be encashed.
A 30-day notice period is required when resigning. Leave during notice is discouraged unless approved.
Managers may approve or reject leave based on workload. Misuse of leave policy may result in disciplinary action.
Employees must ensure proper work handover before taking leave. HR may revise policies based on business needs.
"""

# =========================================
# STREAMLIT UI
# =========================================
st.title("🤖 HR Policy Chatbot")
st.write("Ask questions about the HR leave policy.")

# API key input
api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:
    # =========================================
    # 2. DOCUMENT + SPLIT
    # =========================================
    documents = [Document(page_content=hr_policy, metadata={"source": "internal_policy"})]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # =========================================
    # 3. EMBEDDINGS + FAISS
    # =========================================
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # =========================================
    # 4. SAFE GROQ MODEL LOADER
    # =========================================
    def get_working_llm():
        models = ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma-7b-it"]
        for m in models:
            try:
                llm = ChatGroq(model=m, temperature=0, groq_api_key=api_key)
                llm.invoke("test")  # quick check
                return llm
            except Exception:
                continue
        raise RuntimeError("No working Groq models available.")

    llm = get_working_llm()

    # =========================================
    # 5. PROMPT
    # =========================================
    prompt = PromptTemplate(
        template="""
You are an HR assistant.
- Give clear and concise answers
- Use bullet points when summarizing
- Answer ONLY from the context
- If not found, say: "Not found in HR policy."
Context:
{context}
Question:
{question}
Answer:
""",
        input_variables=["context", "question"]
    )

    # =========================================
    # 6. RAG CHAIN
    # =========================================
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    # =========================================
    # 7. CHAT INTERFACE
    # =========================================
    user_q = st.text_input("Your question:")
    if st.button("Ask"):
        if user_q.strip():
            try:
                res = qa_chain.invoke({"query": user_q})
                st.subheader("🧠 Answer")
                st.write(res["result"])
                st.subheader("📄 Sources")
                for d in res["source_documents"]:
                    st.write("-", d.metadata.get("source"))
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.warning("Please enter your Groq API key to start.")
