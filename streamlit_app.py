import streamlit as st
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


# =========================================
# STREAMLIT PAGE CONFIG
# =========================================
st.set_page_config(page_title="HR Policy Bot", layout="wide")

st.title("🤖 HR Policy Chatbot")


# =========================================
# API KEY INPUT
# =========================================
groq_api_key = st.text_input("Enter your Groq API Key", type="password")

if not groq_api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key


# =========================================
# HR POLICY (IN-MEMORY)
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
# BUILD RAG PIPELINE (CACHE)
# =========================================
@st.cache_resource
def build_rag():

    documents = [
        Document(page_content=hr_policy, metadata={"source": "internal_policy"})
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ✅ MODEL FALLBACK SYSTEM
    def get_llm():
        models = [
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ]

        for m in models:
            try:
                llm = ChatGroq(model=m, temperature=0)
                llm.invoke("test")
                return llm
            except:
                continue

        raise Exception("No working Groq model available")

    llm = get_llm()

    prompt = PromptTemplate(
        template="""
You are an HR assistant.

- Answer ONLY from context
- Be concise
- Use bullet points if needed
- If not found, say: "Not found in HR policy."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain


qa_chain = build_rag()


# =========================================
# CHAT UI
# =========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
query = st.chat_input("Ask about HR policy...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = qa_chain.invoke({"query": query})
            answer = res["result"]

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
