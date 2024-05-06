import os
import shutil
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, GitLoader
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain import hub

from typing import List, Dict

groq_client = None
hf_embeddings = None
LLAMA3_70B = "llama3-70b-8192"
LLAMA3_8B = "llama3-8b-8192"
GEMMA_7B_IT = "gemma-7b-it"

DEFAULT_MODEL = LLAMA3_70B
prev_git_urls = []
vectorStoreRetriever = None


def setup_groq_client(groq_api_key, model_name=DEFAULT_MODEL):
    global groq_client
    groq_client = ChatGroq(temperature=0,
                           model_name=model_name,
                           groq_api_key=groq_api_key)


def load_split_vector(urls: List[str],
                      doc_type="general",
                      file_filter=""):
    global prev_git_urls, hf_embeddings, vectorStoreRetriever
    # Step 1: Load the document from a web url
    if doc_type == "git":
        if os.path.isdir("temp_path") and prev_git_urls[0] != urls[0]:
            shutil.rmtree('temp_path')
        prev_git_urls = urls
        loader = GitLoader(clone_url=urls[0],
                           repo_path="temp_path",
                           file_filter=lambda file_path: file_filter in file_path)
        top_k = 20
    else:
        if prev_git_urls != urls:
            loader = WebBaseLoader(urls)
            prev_git_urls = urls
            top_k = 20
        else:
            return vectorStoreRetriever
    
    documents = loader.load()

    # Step 2: Split the document into chunks with a specified chunk size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    all_splits = text_splitter.split_documents(documents)

    # Step 3: Store the document into a vector store with a specific embedding model
    if not hf_embeddings:
        hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorStore = FAISS.from_documents(all_splits, hf_embeddings)
    vectorStoreRetriever = vectorStore.as_retriever(search_kwargs={'k': top_k})
    return vectorStoreRetriever


def generate_llm_response(chat_history,
                          doc=True,
                          vectorStoreRetriever=None):
    global groq_client
    if doc:
        # Contextualize question
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            groq_client, vectorStoreRetriever, contextualize_q_prompt
        )

        # Answer question
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. Use "
            "the following pieces of retrieved context/document to answer the "
            "question. If you don't know the answer, just say that you "
            "don't know."
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
            ]
        )
        # Below we use create_stuff_documents_chain to feed all retrieved context
        # into the LLM. Note that we can also use StuffDocumentsChain and other
        # instances of BaseCombineDocumentsChain.
        question_answer_chain = create_stuff_documents_chain(groq_client,
                                                             qa_prompt)
        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
        response = rag_chain.invoke({"input": chat_history[-1]["content"],
                                     "chat_history": chat_history})
        return response['answer']
    else:
        return groq_client.invoke(chat_history).content


def groq_chat_completion(urls: List[str],
                         session_messages: List[tuple],
                         doc_type: str = "general",
                         file_filter=""):
    chat_history = session_messages
    if len(urls) > 0:
        vectorStoreRetriever = load_split_vector(urls, doc_type, file_filter)
        response = generate_llm_response(chat_history,
                                         True,
                                         vectorStoreRetriever)
    else:
        response = generate_llm_response(chat_history, False)
    return response
