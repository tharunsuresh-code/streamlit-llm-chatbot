import streamlit as st
import os
from llm import setup_groq_client, groq_chat_completion
import time
import random

# Streamlit app title
st.title("LLAMA3 Chatbot")

os.environ["GROQ_API_KEY"] = st.sidebar.text_input('Groq API Key', type='password')
external_url_source = st.sidebar.text_input('Enter External Source URL:', placeholder='<URL>')

# Streamed response emulator
def response_generator(urls, session_messages, prompt):
    # response = random.choice(
    #     [
    #         "Hello there! How can I assist you today?",
    #         "Hi, human! Is there anything I can help you with?",
    #         "Do you need help?",
    #     ]
    # )
    response = groq_chat_completion(urls, session_messages, prompt)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if not os.environ["GROQ_API_KEY"].startswith('gsk_'):
    st.warning('Please enter your Groq API key!', icon='⚠')

if os.environ["GROQ_API_KEY"].startswith('gsk_'):
    setup_groq_client()

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
    st.session_state.messages.append({"role": "user", "content": prompt})

    urls = []
    if external_url_source:
        urls = external_url_source.split(",")
    with st.chat_message("assistant"):
        # response = st.write_stream(response_generator())
        response = st.write_stream(response_generator(
            urls=urls,
            session_messages=st.session_state.messages,
            prompt=prompt
        ))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    external_url_source = ""
