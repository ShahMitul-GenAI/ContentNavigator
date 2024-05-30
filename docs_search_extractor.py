#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import pickle
import pathlib
from os import listdir
from dotenv import load_dotenv
from os.path import isfile, join
from IPython.display import display, Markdown
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, JSONLoader, TextLoader, UnstructuredFileLoader, DirectoryLoader, \
    Docx2txtLoader, UnstructuredPowerPointLoader

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.chains import RetrievalQA


# In[8]:


# loading API keys from env
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME")


# In[9]:


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

# Get file type
my_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]

# Define a function to create a DirectoryLoader to load text from different formats
# https://github.com/langchain-ai/langchain/discussions/18559
# https://github.com/langchain-ai/langchain/discussions/9605

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


# In[10]:


# gathering file contents

documents = []
for each in my_files:
    file_extension = pathlib.Path(each).suffix
    documents.extend(create_directory_loader(file_extension).load())


# In[11]:


# spliting documents text
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=50)
split_content = text_splitter.split_documents(documents)
print(f"Total {len(split_content)} text chunks created.")


# In[12]:


# Clearning Pinecone index for repetitive useage
def clear_vectorDB(inx):
    index = pc.Index(idx)
    index.delete(
        delete.all==True
    )


# In[13]:


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
    
    clear_vectorDB(INDEX_NAME)
    database = PineConeVectorStore.from_documents(
        documents = documents,
        embedding = embeddings,
        index_name = INDEX_NAME,
    )
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")


# In[14]:


# setting up retreival
qa_ans = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = 'stuff',
    retriever = database.as_retriever(),
    return_source_documents = True,
    verbose=True,
)


# In[17]:


# Develop query for searching documents
# file_path = os.path.dirname(os.path.abspath(__file__))
with open("./docs/user_query.txt", "r", encoding='utf-8') as f:
    question = f.read()
# question = "What are Donald Trump's major achievements as the president?"


# In[29]:


# Testing the model
response = qa_ans(question)

with open('./docs/response_dict.pkl', 'wb') as f:
    pickle.dump(response, f)
print(response['result'])
print(response['source_documents'])

