import os
from os import listdir
from os.path import isfile, join
import pathlib
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader, TextLoader, UnstructuredFileLoader, DirectoryLoader, \
    Docx2txtLoader, UnstructuredPowerPointLoader

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.chains import RetrievalQA

# loading API keys from env
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")

# setting up LLM and Embeddings
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(openai_api_key=str(OPENAI_API_KEY))

# Provide file path & definingn functions  
file_path = "./docs/"

# Define a dictionary to map file extensions to their respective loaders
loaders = {
    '.pdf': PyPDFLoader,
    '.doc': Docx2txtLoader,
    '.docx': Docx2txtLoader,
    '.csv': CSVLoader,
    '.json': JSONLoader,
    '.txt': TextLoader,
    '.htm': UnstructuredFileLoader,
    '.html': UnstructuredFileLoader,
    '.ppt': UnstructuredPowerPointLoader,
    '.pptx': UnstructuredPowerPointLoader,
}

def create_directory_loader(file_type):
    if file_type == '.txt':
        text_loader_kwargs={'autodetect_encoding': True}
        return DirectoryLoader(
            path=file_path,
            glob=f"**/*{file_type}",
            loader_cls=loaders[file_type],
            silent_errors = True,
            loader_kwargs=text_loader_kwargs,
        )
    else:
        return DirectoryLoader(
            path=file_path,
            glob=f"**/*{file_type}",
            loader_cls=loaders[file_type],
            silent_errors = True,
        )


# Clearning Pinecone index for repetitive useage
def clear_vectorDB(pnc_key: str, pnc_inx: str):

    # deleting the existing inex
    pc = Pinecone(api_key=str(pnc_key))
    pc.delete_index(pnc_inx)

    # recreating the deleted index for use
    pc.create_index(
        name = pnc_inx,
        dimension = 1536,
        metric = "cosine",
        spec = ServerlessSpec(
            cloud = "aws",
            region = "us-east-1"
        )
    )


def load_documents():
    # Get file type
    my_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]

    # gathering file contents
    documents = []

    for file in my_files:
        file_extension = pathlib.Path(file).suffix
        documents.extend(create_directory_loader(file_extension).load())

    return documents


def generate_response(search_query, database):

    # setting up retreival
    qa_ans = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = database.as_retriever(),
        return_source_documents = True,
        verbose=False,
    )
    response = qa_ans(search_query)
    return response


def main(query):
    documents = load_documents()

    database = DocArrayInMemorySearch.from_documents(
        documents,
        embeddings,
        )
    response = generate_response(query, database)
    return response

def create_pinecone_database(pnc_key: str, pnc_indx: str):
    
    documents = load_documents()
    pc = Pinecone(api_key=str(pnc_key))

    # clean Pinecone VectorStore for another round of use
    clear_vectorDB(pnc_key, pnc_indx)

    # creating vector database
    database = PineconeVectorStore.from_documents(
        documents = documents,
        embedding = embeddings,
        index_name = pnc_indx
    )
    return database

def pincone_output(query: str, pnc_key: str, pnc_indx: str):
    
    pnc_database = create_pinecone_database(str(PINECONE_API_KEY), str(PINECONE_INDEX))
    response = generate_response(query, pnc_database)    
    return response