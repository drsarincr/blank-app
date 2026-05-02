import streamlit as st
from groq import Groq

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch


# =========================================
# PAGE SETUP
# =========================================
st.set_page_config(page_title="HR Policy Bot")
st.title("🤖 HR Policy Chatbot")


# =========================================
# API KEY INPUT
# =========================================
api_key = st.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.stop()

client = Groq(api_key=api_key)


# =========================================
# HR POLICY (IN-MEMORY)
# =========================================
hr_policy = """
Employees are entitled to 20 days of paid leave annually including casual, sick, and earned leave.
Casual leave is 8 days, sick leave is 7 days with medical proof if more than 2 days.
Earned leave can be carried forward up to 30 days.

Leave must be applied 3 days in advance. Emergency leave must be informed immediately.

Maternity leave is 26 weeks. Paternity leave is 5 days.

Public holidays are separate. Working on holidays gives compensatory off.

Earned leave can be encashed. Casual and sick leave cannot.

Notice period is 30 days. Leave during notice is discouraged.

Managers can approve or reject leave based on workload.
"""


# =========================================
# BUILD RAG PIPELINE
# =========================================
@st.cache_resource
def build_retriever():
    docs = [Document(page_content=hr_policy)]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = DocArrayInMemorySearch.from_documents(split_docs, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 3})


retriever = build_retriever()


# =========================================
# PROMPT TEMPLATE
# =========================================
def build_prompt(context, question):
    return f"""
You are an HR assistant.

Answer ONLY from the context.
If not found, say: Not found in HR policy.

Context:
{context}

Question:
{question}

Answer:
"""


# =========================================
# GROQ LLM CALL
# =========================================
def ask_llm(context, question):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an HR assistant."},
            {"role": "user", "content": build_prompt(context, question)}
        ],
        temperature=0
    )

    return response.choices[0].message.content


# =========================================
# CHAT UI
# =========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
query = st.chat_input("Ask about HR policy...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve relevant chunks
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])

    # Generate answer
    answer = ask_llm(context, query)

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
