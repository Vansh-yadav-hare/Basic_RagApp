import streamlit as st
import tempfile
import os

from rag_pipeline import process_file, ask_question

# ------------------- CONFIG -------------------

st.set_page_config(page_title="RAG App", layout="wide")
st.title("🧠 Chat with Your Data")

# ------------------- SESSION STATE -------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------- SIDEBAR -------------------

with st.sidebar:
    st.header("📂 Upload File")
    uploaded_file = st.file_uploader(
        "Upload PDF / CSV / TXT",
        type=["pdf", "csv", "txt"]
    )

# ------------------- FILE PROCESSING -------------------

if uploaded_file:
    # ✅ Preserve file extension (IMPORTANT FIX)
    file_extension = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    st.info(f"Processing: {uploaded_file.name}")

    with st.spinner("Processing file..."):
        process_file(file_path)

    st.success("✅ File processed successfully!")

# ------------------- CHAT HISTORY -------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ------------------- USER INPUT -------------------

user_query = st.chat_input("Ask something about your data...")

if user_query:
    # User message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Model response
    with st.spinner("Thinking..."):
        answer = ask_question(user_query)

    # Assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)