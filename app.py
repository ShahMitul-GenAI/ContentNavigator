import os
from pathlib import Path
import streamlit as st
from streamlit import session_state as ss 
import pandas as pd
from doc_search.doc_search import main, pincone_output
import shutil

# check for environment file
if not Path("./.env").is_file():
    raise ValueError(st.write("Please create environment file with API keys per README"))

# declaring session_state variables
if 'user_inputs' not in ss:
    ss.user_inputs = ""


# extract source documents info
def get_meatadata(source_docs):
    metadata_info = []
    for each in source_docs:
        doc_meta = each.metadata['source']
        metadata_info.append(doc_meta)
    return set(metadata_info)


# display final results
def display_output(output):
    st.markdown(" ### The search results: \n")
    st.write(output['result'])
    st.markdown(" ### The source documents are: \n")
    st.write(get_meatadata(output["source_documents"]))

# clearing previously loaded pool of docs
target_folder = Path(__file__).resolve().parent / "docs/"

if target_folder.exists():
    shutil.rmtree(target_folder)
target_folder.mkdir(parents=True, exist_ok=True)

with open(target_folder / "user_query.txt", "w") as fp:
    fp.write("N/A")


# function to execute backend python file
st.title("Content Navigator")
st.header("Please select multiple documents for information search")

uploaded_docs = st.file_uploader(
    label="Please upload your files (Total Limit: 200 MB)",
    accept_multiple_files=True,
    type=['pdf', 'doc', 'docx', 'csv', 'json', 'txt', 'htm', 'html', 'ppt', 'pptx']
)

# displaying the user uploaded file(s) and saving to local folder
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

    df_display["Name"] = pd.Series(file_name)
    df_display["Type"] = pd.Series(file_type)
    df_display["Size"] = pd.Series(file_size)
    st.markdown(" **Your uploaded file(s) are: ** ")
    st.dataframe(df_display)

# develop form
with st.form(key="data_extractor", clear_on_submit=False):
    # get the input about search query
    ss.user_inputs = st.text_input(
        label = "Please key-in what you want to search from the pool of above documents (max: 100 characters).",
        max_chars = 250
    )

    # submit action
    submit_button_1 = st.form_submit_button("Submit")

if submit_button_1:
    try:
        with st.spinner("Processing your informatoin..."):
            output = main(ss.user_inputs)
            
            st.success("Data Processing Completed!")
            # displaying output now
            display_output(output)
    
    except ValueError:
        with st.spinner("Processing your request....."):
            output = pincone_output(ss.user_inputs)
            st.success ("Data Processing Complete!")
            display_output(output)