import os
import json
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def scan_folder(folder_path):
    metadata = {}
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if it's a file (not a directory) and a pdf
        if os.path.isfile(file_path) and file_path.endswith(".pdf"):
            document_path = os.path.abspath(file_path)
            mod_time = os.path.getmtime(document_path)
            print(f'> found document at {document_path}')

            metadata[file_path] = {
                "mod_time": mod_time
            }

    return metadata

def process_pdfs(folder_path, vector_store):

    current_metadata = scan_folder(folder_path)

    metadata_file = f'{folder_path}/metadata.json'

    if os.path.exists(metadata_file):
        last_metadata = load_metadata(metadata_file)
    else:
        last_metadata = {}

    changes = detect_changes(current_metadata, last_metadata)
    print("> changes detected:", changes)

    # Add new or modified documents
    if changes["added"] or changes["modified"]:
        for doc in (changes["added"] + changes["modified"]):
            new_doc = load_and_split_document(doc) 
            vector_store.add_documents(new_doc, ids=[d.metadata["source"] for d in new_doc])

    # Delete removed documents
    for file_path in changes["deleted"]:
        vector_store.delete([file_path])

    # Save current metadata for the next scan
    save_metadata(current_metadata, metadata_file)


def load_and_split_document(document_path):
    # PyMuPDFLoader is the fastest of the PDF parsing options,
    # returns one document per page.
    loader = PyMuPDFLoader(document_path)
    document = loader.load()

    print(f'> loaded {len(document)} pages')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )

    document_chunks = text_splitter.split_documents(document)
    print(f'> done processing document {len(document_chunks)} chunks found')

    return document_chunks

def save_metadata(metadata, output_file):
    with open(output_file, "w") as f:
        json.dump(metadata, f)

def load_metadata(input_file):
    with open(input_file, "r") as f:
        return json.load(f)

def detect_changes(current_metadata, last_metadata):
    changes = {
        "modified": [],
        "added": [],
        "deleted": []
    }

    # Check for modified or deleted files
    for file_path, last_data in last_metadata.items():
        if file_path not in current_metadata:
            changes["deleted"].append(file_path)
        else:
            current_data = current_metadata[file_path]
            if current_data["mod_time"] != last_data["mod_time"]:
                changes["modified"].append(file_path)

    # Check for added files
    for file_path in current_metadata:
        if file_path not in last_metadata:
            changes["added"].append(file_path)

    return changes

def get_or_create_embeddings(folder_path, model, collection_name):

    embeddings = OllamaEmbeddings(model=model)

    print('> initializing vector store')
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="../.chroma_db",
    )

    process_pdfs(folder_path, vector_store)

    retriever = vector_store.as_retriever()

    return retriever


def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)