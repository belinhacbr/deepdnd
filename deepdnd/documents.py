import os
import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import ollama

def load_pdfs(folder_path):
    documents = []

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if it's a file (not a directory) and a pdf
        if os.path.isfile(file_path) and file_path.endswith(".pdf"):
            document_path = os.path.abspath(file_path)
            print(document_path)

            # PyMuPDFLoader is the fastest of the PDF parsing options,
            # returns one document per page.
            loader = PyMuPDFLoader(document_path)
            documents.extend(loader.load())

    return documents


def process_pdfs(folder_path):

    documents = load_pdfs(folder_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )

    document_chunks = text_splitter.split_documents(documents)
    print(document_chunks[0])
    print(document_chunks[-1])

    return document_chunks


def make_embeddings(document_chunks, model):
    embeddings = OllamaEmbeddings(model=model)
    vectorstore = Chroma.from_documents(
        documents=document_chunks, embedding=embeddings, persist_directory="../.chroma_db"
    )
    retriever = vectorstore.as_retriever()

    return retriever


def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)