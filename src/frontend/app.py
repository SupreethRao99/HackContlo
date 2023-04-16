import sys
sys.path.extend(["./", "../", ".../"])

import os
print(os.listdir())

import time
import streamlit as st
import pandas as pd
import boto3
from botocore.exceptions import NoCredentialsError
import requests
from src.backend.models.auto_segment import AutoSegmentor
from src.backend.models.persona_generation import PersonaGenerator
from src.backend.models.strategy_generation import StrategyGenerator

st.set_page_config(page_title="ContloHack", page_icon=":sparkles:", layout="wide")


st.title("Upload CSV File")

# Allow user to upload a file
uploaded_file = st.file_uploader("Choose an CSV file", type=["csv"])

def process(df):

    st.table(df.head(5))
    segmentor = AutoSegmentor(df)

    progress_text = "Data getting Clustered. Please wait..."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        my_bar.progress(percent_complete+1)
        time.sleep(0.1)

    clustered_segments = segmentor.get_output()

    st.success('Data Clustering DONE !', icon="✅")

    persona_gen = PersonaGenerator()
    strategy_gen = StrategyGenerator()

    with st.spinner("Generating Personas and Possible stratergies for every persona ... ") :

        for cluster in clustered_segments:
            persona = persona_gen.get_user_persona(
                segmentation_type=cluster["cluster_data"], 
                attributes=cluster["segment_attributes"], 
                cluster_data=cluster["cluster_data"]
            )
            st.subheader(persona)
            # st.table(pd.DataFrame(cluster["cluster_data"]))

            
            retention_strategy = strategy_gen.get_retention_strategy(
                user_persona=persona, 
                segmention_type=cluster["segment_type"]
            )
            st.write(retention_strategy)
            
            time.sleep(3)


if uploaded_file is not None:
    # Load the file into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Ask the user if they want to select specific columns or use all columns
    select_option = st.radio("Select an option", ("Select specific columns", "Use all columns"), key="1")

    # If the user wants to select specific columns
    if select_option == "Select specific columns":
        # Display a list of column names
        col_names = df.columns.tolist()
        selected_cols = st.multiselect("Select columns", col_names)

        # If the user has selected some columns
        if selected_cols:
            # Filter the DataFrame to include only the selected columns
            df = df[selected_cols]            
            process(df)
    
    else:

        process(df)




            

