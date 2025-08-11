# app.py
import streamlit as st
from query_agent import generate_answer

st.set_page_config(page_title="Internal Knowledge Assistant", layout="centered")

st.title("📚 Internal Knowledge Assistant")
st.markdown("Ask any question based on your internal documents.")

query = st.text_input("🔍 Enter your question:")

if query:
    with st.spinner("Generating answer..."):
        answer = generate_answer(query)
        st.markdown("### 💬 Answer")
        st.write(answer)
