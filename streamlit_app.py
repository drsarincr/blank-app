import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="HR Policy Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 HR Policy Chatbot")
st.caption("Powered by Groq LLM + FAISS + LangChain")

# =========================================
# SIDEBAR — API KEY INPUT
# =========================================
with st.sidebar:
    st.header("🔑 Configuration")
    groq_api_key = st.text_input(
        "Enter your Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free API key at https://console.groq.com"
    )
    st.markdown("---")
    st.markdown("**How to get a Groq API Key:**")
    st.markdown("1. Go to [console.groq.com](https://console.groq.com)")
    st.markdown("2. Sign up / Log in")
    st.markdown("3. Navigate to API Keys")
    st.markdown("4. Create and copy your key")
    st.markdown("---")
    st.markdown("**Models tried (auto-fallback):**")
    st.markdown("- llama-3.1-8b-instant")
    st.markdown("- mixtral-8x7b-32768")
    st.markdown("- gemma-7b-it")

# =========================================
# HR POLICY TEXT
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
# CACHED SETUP (only re-runs if policy changes)
# =========================================
@st.cache_resource(show_spinner="📚 Building vector store...")
def build_retriever():
    documents = [Document(page_content=hr_policy, metadata={"source": "internal_policy"})]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# =========================================
# LLM LOADER WITH AUTO-FALLBACK
# =========================================
def get_working_llm(api_key: str):
    models = [
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma-7b-it"
    ]
    for model in models:
        try:
            llm = ChatGroq(model=model, temperature=0, groq_api_key=api_key)
            llm.invoke("test")
            return llm, model
        except Exception:
            continue
    raise RuntimeError("❌ No working Groq models available. Please check your API key.")

# =========================================
# PROMPT TEMPLATE
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
# MAIN CHAT UI
# =========================================
if not groq_api_key:
    st.info("👈 Please enter your **Groq API Key** in the sidebar to start chatting.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Build retriever (cached)
retriever = build_retriever()

# Load LLM
if "llm_model" not in st.session_state or "llm" not in st.session_state:
    with st.spinner("🔌 Connecting to Groq..."):
        try:
            llm, model_name = get_working_llm(groq_api_key)
            st.session_state.llm = llm
            st.session_state.llm_model = model_name
            st.success(f"✅ Connected using model: `{model_name}`")
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

# Build QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=st.session_state.llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask about HR policies... (e.g. How many sick leaves do I get?)"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke({"query": user_input})
                answer = result["result"]
                sources = list({d.metadata.get("source", "unknown") for d in result["source_documents"]})

                st.markdown(answer)
                st.caption(f"📄 Source: `{', '.join(sources)}`")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
            except Exception as e:
                err_msg = f"❌ Error: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})

# Clear chat button
if st.session_state.messages:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
