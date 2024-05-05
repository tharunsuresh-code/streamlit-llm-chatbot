import os
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

from typing import List, Dict

groq_client = None
LLAMA3_70B = "llama3-70b-8192"
LLAMA3_8B = "llama3-8b-8192"

DEFAULT_MODEL = LLAMA3_70B


def setup_groq_client():
    global groq_client
    # groq_client = Groq(
    #     api_key=os.environ.get("GROQ_API_KEY"),
    # )
    groq_client = ChatGroq(temperature=0, model_name=DEFAULT_MODEL)


# def assistant(content: str):
#     return {"role": "assistant", "content": content}


# def user(content: str):
#     return {"role": "user", "content": content}


# def chat_completion(
#     messages: List[Dict],
#     model=DEFAULT_MODEL,
#     temperature: float = 0.6,
#     top_p: float = 0.9,
# ) -> str:
#     response = groq_client.chat.completions.create(
#         messages=messages,
#         model=model,
#         temperature=temperature,
#         top_p=top_p,
#     )
#     return response.choices[0].message.content


def load_split_vector(urls: List[str]):
    # Step 1: Load the document from a web url
    loader = WebBaseLoader(urls)
    documents = loader.load()

    # Step 2: Split the document into chunks with a specified chunk size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                   chunk_overlap=0)
    all_splits = text_splitter.split_documents(documents)

    # Step 3: Store the document into a vector store with a specific embedding model
    vectorStore = FAISS.from_documents(all_splits,
                                       HuggingFaceEmbeddings(
                                           model_name="sentence-transformers/all-mpnet-base-v2"))
    return vectorStore


def groq_chat_completion(urls: List[str], 
                         session_messages: List[tuple],
                         prompt: str):
    if len(urls) > 0:
        vectorStore = load_split_vector(urls)
        chain = ConversationalRetrievalChain.from_llm(groq_client,
                                                      vectorStore.as_retriever(),
                                                      return_source_documents=True)
    else:
        chain = ConversationalRetrievalChain.from_llm(groq_client,
                                                      return_source_documents=True)

    chat_history = []
    for i in range(0, len(session_messages)-1, 2):
        chat_history.append((f"User: {session_messages[i]['content']}", 
                             f"Assistant: {session_messages[i+1]['content']}"))
    print(chat_history)
    result = chain.invoke({"question": prompt, "chat_history": chat_history})
    return result['answer']