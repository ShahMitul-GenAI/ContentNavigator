import os.path
import time
import tiktoken
from pathlib import Path
import pickle
import streamlit as st
from streamlit import session_state as ss
import pandas as pd
from subprocess import call
# from doc_search import search_results_extdb, search_results_memory
from doc_search import search_results_memory

# sync folder to store the uploaded documents
target_folder = Path(str(Path().absolute()) + "/docs/")

# clearing previously loaded pool of docs
delete_files = [f.unlink() for f in Path(str(target_folder)).iterdir() if f.is_file()]
del delete_files

# getting context token size for VectorStore selection 
def count_tokens(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# display final results
def display_output(output):
    st.markdown(" ### The search results from the uploaded documents is: \n")
    st.write(output['result'])
    st.markdown(" ### The source documents of the above search results are: \n")
    st.write(output['source_documents'])

# re-setting the session stage for nested form buttons
def count(key):
    ss[key] += 1

# User Interface 
st.title("Content Search in Multiple Documents")
st.header("Provide the requested information")

# declaring  session_state variables
if 'user_inputs' not in ss:
    ss.user_inputs = ""
if 'pnc_key' not in ss:
    ss.pnc_key = ""
if 'pnc_index' not in ss:
    ss.pnc_index = ""
if 'sb1_count' not in ss:
    ss.sb1_count = 0
if "sb2_count" not in ss:
    ss.sb2_count = 0

uploaded_docs = st.file_uploader(
    label="Please upload your files (Limit 100 MB per file, max 15 files)",
    accept_multiple_files=True,
    type=['pdf', 'doc', 'docx', 'csv', 'json', 'txt', 'htm', 'html', 'ppt','pptx']
)

if uploaded_docs:
        df_display = pd.DataFrame()

        file_name = []
        file_type = []
        file_size = []

        for uploaded_file in uploaded_docs:
            file_name.append(uploaded_file.name)
            file_type.append(uploaded_file.type.split(".")[-1])
            file_size.append(uploaded_file.size)
            with open(os.path.join(target_folder, uploaded_file.name), 'wb') as f:
                f.write(uploaded_file.getbuffer())

        df_display["No."] = [(i+1) for i in range(len(file_name))]
        df_display["File Name"] = pd.Series(file_name)
        df_display["File Type"] = pd.Series(file_type)
        df_display["File_size"] = pd.Series(file_size)
        st.markdown(" **Your uploaded file(s) are: ** ")
        st.dataframe(df_display)

# main form
with st.form(key="data_extractor", clear_on_submit=True):
    # get the input about search query
    ss.user_inputs = st.text_input(
        label = "Please key-in what you want to search from the pool of above documents (max: 100 characters).",
        max_chars = 100
    )
    
    # submit action 

    submit_button1 = st.form_submit_button("Submit", on_click=count, args=("sb1_count",))

# keep the submit button 1 on 
submit_button1 = bool(ss.sb1_count % 2)

if submit_button1:
    
    # export_inputs(ss.user_inputs)

    # check for external vectorstore
    # documents = get_documents()
    # content = get_text(documents)
    # total_tokens = count_tokens(str(content), "cl100k_base")

    # try:
    with st.spinner("Processing your request....."):
        output = search_results_memory(ss.user_inputs)
        st.success ("Data Processing Complete!")
        display_output(output)

    # except:
    #     st.write("The content pool is large enough to be handled by the local database.")
    #     st.write("Pinecone Vector Database will be used for embedding & retrieval")

    #     # get credentials to run external vectorstore
    #     with st.form(key="VectorDB", clear_on_submit=False):
    #         ss.pnc_key  = st.text_input(
    #         label = "Please input your Pinecone API Key",
    #                 max_chars = 150,
    #                 type ="password")

    #         ss.pnc_index = st.text_input(
    #                 label = "Plese input your Pinecone Index Name",
    #                 max_chars = 150)
    #         submit_button2 = st.form_submit_button("Submit Pinecone Credentials", on_click=count, args=("sb2_count",))

    #     # keep submit button 2 on
    #     submit_button2 = bool(ss.sb2_count % 2)

    #     if submit_button2:
    #         with st.spinner("Processing your request....."):
    #             print("I am now in submit button 2")
    #             output = search_results_extdb(ss.user_inputs, ss.pnc_key, ss.pnc_index)
    #             st.success ("Data Processing Complete!")
    #             display_output(output)


















