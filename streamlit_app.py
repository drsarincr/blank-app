import streamlit as st
from groq import Groq

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="HR Policy Bot")
st.title("🤖 HR Policy Chatbot")

# =========================================
# API KEY INPUT
# =========================================
api_key = st.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.warning("Enter your Groq API key to continue")
    st.stop()

client = Groq(api_key=api_key)

# =========================================
# HR POLICY (LONG TEXT)
# =========================================
hr_policy = """
The company provides employees with 20 days of paid leave annually, including casual, sick, and earned leave.
Casual leave is limited to 8 days per year and is used for personal reasons.
Sick leave is 7 days per year and requires medical proof if taken for more than 2 consecutive days.
Earned leave accumulates over time and can be carried forward up to 30 days.

Employees must apply for leave at least 3 days in advance.
Emergency leave must be communicated immediately.

Maternity leave is 26 weeks as per law.
Paternity leave is 5 days.

Public holidays are separate from leave balances.
Working on holidays provides compensatory leave.

Earned leave can be encashed during resignation.
Casual and sick leave cannot be encashed.

A notice period of 30 days is required when resigning.
Leave during the notice period is discouraged.

Managers may approve or reject leave based on business needs.
Misuse of leave policy can result in disciplinary action.
"""

# =========================================
# CHAT MEMORY
# =========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================================
# USER INPUT
# =========================================
query = st.chat_input("Ask about HR policy...")

if query:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # =========================================
    # GROQ RESPONSE (NO LANGCHAIN)
    # =========================================
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an HR assistant. Answer only using the given policy."
                    },
                    {
                        "role": "user",
                        "content": f"""
HR POLICY:
{hr_policy}

QUESTION:
{query}

Answer clearly and concisely.
If not found, say: Not found in HR policy.
"""
                    }
                ]
            )

            answer = response.choices[0].message.content

            st.markdown(answer)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
