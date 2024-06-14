import streamlit as st
import os
from llm import (setup_groq_client, groq_chat_completion,
                 LLAMA3_70B, LLAMA3_8B, GEMMA_7B_IT)
import time


st.set_page_config(
        page_title="RAG Chatbot",
)


# Streamlit app title
st.title("RagyBot - A RAG based chatbot")

backend_LLM = LLAMA3_70B
file_filter = None
GROQ_API_KEY = None
uploaded_file = None
external_url_source = None


def setup_groq_with_backend():
    if not GROQ_API_KEY or not GROQ_API_KEY.startswith('gsk_'):
        st.warning('Please enter your Groq API key!', icon='⚠')
    else:
        setup_groq_client(GROQ_API_KEY, backend_LLM)


st.sidebar.text("❤️ Built with love by Tharun Suresh")

GROQ_API_KEY = st.sidebar.text_input('Groq API Key',
                                     type='password')
backend_LLM = st.sidebar.selectbox("LLM",
                                   options=(LLAMA3_70B, LLAMA3_8B,
                                            GEMMA_7B_IT),
                                   on_change=setup_groq_with_backend())
doc_type = st.sidebar.selectbox("Doc Type", options=("general", "git", "pdf"))

if doc_type != "pdf":
    external_url_source = st.sidebar.text_input('Enter External Source URL:',
                                                placeholder='<URL>')


# Streamed response emulator
def response_generator(urls,
                       session_messages,
                       doc_type,
                       file_filter=file_filter,
                       uploaded_file=uploaded_file):
    response = groq_chat_completion(urls, session_messages, doc_type,
                                    file_filter=file_filter,
                                    uploaded_file=uploaded_file)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# if not os.environ["GROQ_API_KEY"].startswith('gsk_'):
#     st.warning('Please enter your Groq API key!', icon='⚠')

if GROQ_API_KEY and GROQ_API_KEY.startswith('gsk_'):
    if "set_groq" not in st.session_state:
        st.balloons()
        st.toast("You are all set to use the chatbot!", icon='✅')
        setup_groq_with_backend()
        st.session_state.set_groq = True

if doc_type == "git":
    file_filter = st.sidebar.text_input('File filter:',
                                        placeholder='Enter file type/name to filter the git repo')
    if not file_filter:
        st.sidebar.text('Note: Please filter files to speed \nthe context!')

if doc_type == "pdf":
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    if GROQ_API_KEY and GROQ_API_KEY.startswith('gsk_'):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user",
                                          "content": prompt})

        urls = []
        if external_url_source:
            urls = external_url_source.split(",")
        assistant_message = st.chat_message("assistant")
        with st.spinner("Thinking..."):
            llm_response = response_generator(
                urls=urls,
                doc_type=doc_type,
                session_messages=st.session_state.messages,
                file_filter=file_filter,
                uploaded_file=uploaded_file
            )
            response = assistant_message.write_stream(llm_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant",
                                          "content": response})
        external_url_source = ""
    else:
        st.toast("Please enter your Groq API key!", icon='❗')
