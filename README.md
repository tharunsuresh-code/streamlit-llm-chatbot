# streamlit-llm-chatbot
A simple chatbot using Groq backed LLM using Retrieval Augmented Generation (RAG) with Langchain. 

[<img src="https://cdn.loom.com/sessions/thumbnails/e8d4c852300d46678484d22b9debb49a-with-play.gif" width="600" height="300"
/>](https://www.loom.com/share/e8d4c852300d46678484d22b9debb49a)


It is a streamlit app that lets you make retrieval augmented generation from a web source using LLM and [LangChain](https://github.com/langchain-ai/langchain). It is supported by [Groq](https://groq.com/) API to make the LLM inference.(It is free for personal use with limits, highly recommended to try it out!)

What can be done?

1. Connect to a [Llama3](https://github.com/meta-llama/llama3) or [Gemma](https://github.com/google/gemma_pytorch) using your Groq API key, and chat with it!
2. Input any website, and query/chat based on questions from the website and LLM's knowledge.
3. Input a Git repository, and query/chat based on the files in the repository.

Try out the app on Streamlit - [streamlit-llm-chatbot](https://groq-llm-langchain-chatbot.streamlit.app/)

## Setup instructions
Create a virtual environment to install from requirements file. You need Python 3.10 or higher.

```
pip install -r requirements.txt
```

## Run
To test the app against the Churchill insurance document:
```
python test.py
```

## Credits
A big shoutout to all the detailed documentations from [Langchain](https://python.langchain.com/docs/get_started/introduction) and [Streamlit](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps) to help build the app. [Groq](https://groq.com/) has been generous with their rate limits for the LLMs which enabled the development of the app. Thanks to [pip-chill](https://github.com/rbanffy/pip-chill) for a chill pip freeze. Used [Loom](https://www.loom.com/) for the demo video of the app.
