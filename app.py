import os
import time
from pathlib import Path
import pickle
import streamlit as st
import pandas as pd
from docs_search_extractor import main


# function to export user inputs from form
def export_inputs(data):
    if type(data) == str:
        st.session_state.user_inputs = data
        with open(str(target_folder) + "user_query.txt", "w", encoding='utf-8') as fp:
            fp.write(st.session_state.user_inputs)
    elif type(data) == list:
        st.session_state.credentials = data[:]
        with open(str(target_folder) + "pnc_vals.pkl", "wb") as fb:
            pickle.dump(st.session_state.credentials, fb)


# clearing previously loaded pool of docs
target_folder = str(os.path.dirname(os.path.abspath(__file__))) + "/docs/"
delete_files = [f.unlink() for f in Path(str(target_folder)).iterdir() if f.is_file()]
del delete_files

# function to execute backend python file
st.title("User inputs for content search in multiple documents")
st.header("Provide the requested information")

uploaded_docs = st.file_uploader(label="Please upload your files (Limit 100 MB per file, max 15 files)",
                                    accept_multiple_files=True,
                                    type=['pdf', 'doc', 'docx', 'csv', 'json', 'txt', 'htm', 'html', 'ppt',
                                          'pptx']
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

# declaring  session_state variables
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = ""
if'credentials' not in st.session_state:
    st.session_state['credentials'] = []

# develop form
with st.form(key="data_extractor", clear_on_submit=False):
    # get the input about search query
    st.session_state.user_inputs = st.text_input(
        label = "Please key-in what you want to search from the pool of above documents (max: 100 characters).",
        max_chars = 100
    )

    # submit action
    submit_button_1 = st.form_submit_button("Submit")

if submit_button_1:
    export_inputs(st.session_state.user_inputs)

    query = st.session_state.user_inputs
    print(f"Length of query: {len(query)}")
    print(f"'{query}'")

    main()

    # processing file
    file_check = target_folder + str("response_dict.pkl")

    with st.spinner("Processing your request....."):
        while not os.path.exists(file_check):
            time.sleep(3)
            if os.path.exists(str(target_folder) + "LARGE.txt"):
                pnc = []
                st.write("Your document content pool is large enough to be handled by the local database. "
                         "We must use Pinecone vector database for embedding and retrieval")
                with st.form(key="VectorDB", clear_on_submit=True):
                    pnc_key  = st.text_input(
                        label = "Please input your Pinecone API Key",
                        max_chars = 100,
                        type = password
                    )

                    pnc_index = st.text_input(
                        label = "Plese input your Pinecone Index Name",
                        max_char = 150
                    )

                    pnc.extend([pnc_key, pnc_index])
                    st.session_state.credentials.append(pnc)
                    submit_button_2 = st.form_submit_button("Submit Info")

                if submit_button_2:
                    export_inputs(st.session_state.credentials)
    st.success("Data Processing Complete!")

    with open(str(target_folder) + 'response_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    st.markdown(" **Your requested content is:** ")
    st.write(loaded_dict['result'])
    st.markdown(" \n**The source documents are: ** ")
    st.write(loaded_dict['source_documents'])
