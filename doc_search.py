import os
import pickle
import pathlib
from os import listdir
from dotenv import load_dotenv
from os.path import isfile, join
from typing import Literal, get_args
from IPython.display import display, Markdown
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader, TextLoader, UnstructuredFileLoader, DirectoryLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

file_path = "./docs/"

FileExtensions = Literal[".pdf", ".doc", ".docx", ".csv", '.json', '.txt', '.htm', '.html', '.ppt', '.pptx']
SUPPORTED_EXTENSIONS = get_args(FileExtensions)

# loading API keys from env
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
# INDEX_NAME = os.environ.get("INDEX_NAME")

# pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# setting up LLM and Embeddings
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# set up folder to store the uploaded documents
folder_path = "./docs/"

# defining list of file types recognition mapping
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
  

# directory loader to map all uploded files
def create_directory_loader(file_type: str):
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
    pc = Pinecone(api_key=pnc_key)
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

def get_documents():

    print("I am getting documents now")

    # get list of files uploaded
    my_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    # gathering file contents
    documents = []
    for each in my_files:
        file_extension = pathlib.Path(each).suffix
        print(file_extension)
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"The uploaded file type {file_extension} is not withing the permissible extensions")
        documents.extend(create_directory_loader(file_extension).load())

    # spliting documents text

    return documents    # return a lis

def get_text(documents):

    print("I am getting text now")

    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=50)
    split_content = text_splitter.split_documents(documents)
    content = " ".join([t.page_content for t in split_content])
    return content

def create_memory_database():

    documents = get_documents()
    print("I got documents now")  

    # creating vector database
    database = DocArrayInMemorySearch.from_documents(
        documents = documents,
        embedding = embeddings,
    )
        
    return database   # return a directory 

def create_pinecone_database(pnc_key: str, pnc_indx: str):
    
    print("I am now working on Pinecone")
    
    documents = get_documents()
    pc = Pinecone(api_key=str(pnc_key))

    # clean Pinecone VectorStore for another round of use
    # clear_vectorDB(pnc_key, pnc_indx)

    # creating vector database
    database = PineconeVectorStore.from_documents(
        documents = documents,
        embedding = embeddings,
        index_name = pnc_indx
    )
    return database # return a directory


def retrieve_searches(database, query: str):
    qa_ans = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = database.as_retriever(),
        return_source_documents = True,
        verbose=True,
    )
    results = qa_ans.invoke(query)

    print("I have retrieved the results now")
    return results 

def search_results_memory(query: str):
    
    print("I am in search results function now")
    
    database_package = create_memory_database()
    response = retrieve_searches(database_package, query)
    
    print("Final output received successfully in main file")
    
    return response     # return a directory    

def search_results_extdb(query: str, pnc_key: str, pnc_indx: str):
    
    print("I am in search results function now")
    
    database_package = create_pinecone_database(pnc_key, pnc_indx)
    response = retrieve_searches(database_package, query)
    
    print("Final output received successfully in main file")
    
    return response     # return a directory   