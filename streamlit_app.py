import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Load HR Policy file
DATA_PATH = "HR Policy"
try:
    loader = TextLoader(DATA_PATH)
    documents = loader.load()
except Exception:
    from langchain_core.documents import Document
    documents = [Document(
        page_content="Employees get 20 days leave. Notice period is 30 days.",
        metadata={"source": "demo"}
    )]

# Split + Embed
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = splitter.split_documents(documents)
docs = [d for d in docs if d.page_content.strip() != ""]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")
llm = HuggingFacePipeline(pipeline=generator)

# Prompt
prompt = PromptTemplate(
    template="""
You are an HR assistant.

Answer the question based ONLY on the context below.
- Write a clear, short summary (2–3 sentences).
- Do not repeat or list the context.
- If the answer is not in the context, reply exactly: "Not found in HR policy."

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

# Guard function
def clean_answer(question, result):
    answer = result.strip()
    if len(answer.split()) > 80 or not any(word.lower() in answer.lower() for word in question.split()[:3]):
        return "Not found in HR policy."
    return answer

# Streamlit UI
st.title("🤖 HR Chatbot")

user_input = st.text_input("Ask a question about HR policy:")

if user_input:
    res = qa_chain.invoke({"query": user_input})
    answer = clean_answer(user_input, res["result"])

    st.markdown("### 🧠 Answer")
    st.write(answer)

    st.markdown("### 📄 Sources")
    for d in res["source_documents"]:
        st.write("-", d.metadata.get("source"))
