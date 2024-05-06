import streamlit as st
import os
from llm import setup_groq_client, groq_chat_completion, LLAMA3_70B, LLAMA3_8B, GEMMA_7B_IT
import time


# Streamlit app title
st.title("LLAMA3 Chatbot")

backend_LLM = LLAMA3_70B
file_filter = None

def setup_groq_with_backend():
    if not os.environ["GROQ_API_KEY"].startswith('gsk_'):
        st.warning('Please enter your Groq API key!', icon='⚠')
    else:
        setup_groq_client(backend_LLM)

os.environ["GROQ_API_KEY"] = st.sidebar.text_input('Groq API Key', 
                                                   type='password')
backend_LLM = st.sidebar.selectbox("LLM", options=(LLAMA3_70B, LLAMA3_8B, GEMMA_7B_IT),
                                   on_change=setup_groq_with_backend())
doc_type = st.sidebar.selectbox("Doc Type", options=("general", "git"))
external_url_source = st.sidebar.text_input('Enter External Source URL:',
                                            placeholder='<URL>')


# Streamed response emulator
def response_generator(urls, session_messages, doc_type, file_filter):
    response = groq_chat_completion(urls, session_messages, doc_type, file_filter)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# if not os.environ["GROQ_API_KEY"].startswith('gsk_'):
#     st.warning('Please enter your Groq API key!', icon='⚠')

if os.environ["GROQ_API_KEY"].startswith('gsk_'):
    setup_groq_with_backend()

if doc_type == "git":
    file_filter = st.sidebar.text_input('File filter:',
                                        placeholder='Enter file type/name to filter the git repo')
    if not file_filter:
        st.warning('Please filter files to speed the context!', icon='⚠')


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", 
                                      "content": prompt})

    urls = []
    if external_url_source:
        urls = external_url_source.split(",")
    with st.chat_message("assistant"):
        # response = st.write_stream(response_generator())
        response = st.write_stream(response_generator(
            urls=urls,
            doc_type=doc_type,
            session_messages=st.session_state.messages,
            file_filter=file_filter
        ))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant",
                                      "content": response})
    external_url_source = ""
