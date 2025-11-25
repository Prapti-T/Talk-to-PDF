import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000" 

st.set_page_config(page_title="Talk-to-PDF", layout="centered")
st.title("Talk-to-PDF")

task = st.sidebar.selectbox("Select Task", ["Chat Mode (QA)", "Predict [MASK] (MLM)"])

if task == "Chat Mode (QA)":
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("Ask a question:")

    if st.button("Send") and user_input:
        with st.spinner("Retrieving answer..."):
            try:
                payload = {
                    "query": user_input,
                    "session_id": st.session_state.session_id,
                    "top_k": 5
                }
                response = requests.post(f"{API_URL}/chat", json=payload)
                response.raise_for_status()
                data = response.json()

                # Update session info
                st.session_state.session_id = data["session_id"]
                st.session_state.history = data["history"]

            except Exception as e:
                st.error(f"Error: {e}")

    # Display conversation history
    st.subheader("Conversation History")
    for turn in st.session_state.history:
        role = turn.get("role", "Unknown").capitalize()
        content = turn.get("content", "")
        st.markdown(f"**{role}:** {content}")

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.session_id = None
        st.session_state.history = []


elif task == "Predict [MASK] (MLM)":
    text = st.text_area("Enter text containing [MASK]:", "AI is the [MASK] of machines.")

    if st.button("Predict [MASK]") and "[MASK]" in text:
        with st.spinner("Predicting masked token..."):
            try:
                response = requests.post(
                    f"{API_URL}/mlm",
                    json={"text": text}
                )
                response.raise_for_status()
                data = response.json()

                st.subheader("Predicted Word:")
                st.write(data["predicted_word"])
                st.subheader("Filled Text:")
                st.write(data["filled_text"])
            except Exception as e:
                st.error(f"Error: {e}")
    elif st.button("Predict [MASK]"):
        st.warning("Your text must contain a [MASK] token.")
