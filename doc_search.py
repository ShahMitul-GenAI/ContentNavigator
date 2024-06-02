import os
from os import listdir
from os.path import isfile, join
import pickle
import pathlib
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader, TextLoader, UnstructuredFileLoader, DirectoryLoader, \
    Docx2txtLoader, UnstructuredPowerPointLoader

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.chains import RetrievalQA

# loading API keys from env
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME")

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
def clear_vector_database(pc, inx):
    index = pc.Index(inx)
    index.delete(
        delete.all==True
    )


def load_documents():
    # Get file type
    my_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    print(my_files)

    # Define a function to create a DirectoryLoader to load text from different formats
    # https://github.com/langchain-ai/langchain/discussions/18559
    # https://github.com/langchain-ai/langchain/discussions/9605

    # gathering file contents
    documents = []

    for file in my_files:
        file_extension = pathlib.Path(file).suffix
        documents.extend(create_directory_loader(file_extension).load())

    return documents


def generate_response(search_query, database):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    # setting up retreival
    qa_ans = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = database.as_retriever(),
        return_source_documents = True,
        verbose=True,
    )

    # Develop query for searching documents
    # with open("./docs/user_query.txt", "r", encoding='utf-8') as f:
    #     query = f.read()

    # assert passed_query == query

    # Testing the model
    response = qa_ans(search_query)

    return response


def main(query):
    documents = load_documents()

    # storing the content In-Memory Vector Store 
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    try:
        database = DocArrayInMemorySearch.from_documents(
            documents,
            embeddings,
        )
        print("The content pool is small enough to be handled by the local database is used to handle the query")
    except:
        print("The content pool is large enough to be handled by the local database. \n")
        print("Pinecone Vector Database is now used for embedding & retrieval")
        
        flrg = open(str(file_path) + "LARGE.txt", "wb")
        pickle.dump(123, flrg)
        
        # getting Pinecone credentials from the server
        pnc = []
        pnc = pd.read_pickle(str(file_path) + "pnc_vals.pkl")

        if len(pnc) > 0:
            pc = Pinecone(api_key=os.environ.get(pnc[0]))
            INDEX_NAME = pnc[1]
        else:
            pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
            INDEX_NAME = INDEX_NAME
        
        clear_vector_database(pc, INDEX_NAME)
        database = PineConeVectorStore.from_documents(
            documents = documents,
            embedding = embeddings,
            index_name = INDEX_NAME,
        )

    response = generate_response(query, database)

    print(response['result'])
    print(response['source_documents'])

    return response

if __name__ == "__main__":
    main()