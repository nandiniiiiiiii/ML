import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from src.mcqgenerator.utils import read_file,get_table_data
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.MCQGenerator import generate_chain
from src.mcqgenerator.logger import logging

with open('D:\ML\Generative_AI\Project1\Response.json','r') as file:
    RESPONSE_JSON = json.load(file)

st.title("MCQ Generator")

with st.form("user_input"):
    upload_file= st.file_uploader("Upload a pdf pls")
    mcq_count = st.number_input("No. of MCQ",min_value= 3,max_value=50)
    subject = st.text_input("Enter a subject")
    tone = st.text_input("Enter conplexity of question",max_chars=20,placeholder="Simple")
    button = st.form_submit_button("Create MCQ")

    if button and upload_file is not None and mcq_count and subject and tone:
        with st.spinner("Loading...."):
            try:
                text = read_file(upload_file)
                with get_openai_callback() as cb:
                    response = generate_chain(
                        {
                            "text":text,
                            "number":mcq_count,
                            "subject":subject,
                            "tone":tone,
                            "response_json":json.dumps(RESPONSE_JSON)
                        }
                    )
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("Error")
            else:
                print(f"Total tokens:{cb.total_tokens}")
                print(f"Prompt tokens:{cb.prompt_tokens}")
                print(f"Completion tokens:{cb.completion_tokens}")
                print(f"Total tokens:{cb.total_cost}")
                if isinstance(response,dict):
                    quiz = response.get("quiz",None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df=pd.DataFrame(table_data)
                            df.index = df.index+1
                            st.table(df)
                            st.text_area(label="Review",value=response["review"])
                        else:
                            st.error("Error in the table data")
                else:
                    st.write(response)
            
