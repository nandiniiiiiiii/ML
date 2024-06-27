import os
import pandas as pd
from dotenv import load_dotenv
import langchain
from src.mcqgenerator.logger import logging
from src.mcqgenerator.utils import read_file,get_table_data

from langchain_community.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

load_dotenv()

KEY = os.getenv("OPEN_AI_KEY")

llm=ChatOpenAI(openai_api_key=KEY,model_name="gpt-3.5-turbo",temperature=0.7)

TEMPLATE= """
Text:{text}
You are an expert M maker. Given the above text, it is your job to\
    create a quize of {number} multiple choice questions for {subject} students in {tone} tone.
    Make sure the questions are not repeated and check all the questions to be conforming the text as well.
    Make sure to forma your response like RESPONSE_JSON bellow and use it as a guid.\
    Ensure to make {number} MCQ
    ### RESPONSE_JSON
    {response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables = ["text","number","subject","tone","response_json"],
    template = TEMPLATE
)

quiz_chain = LLMChain(llm=llm,prompt=quiz_generation_prompt,output_key="quiz",verbose=True)

TEMPLATE2= """
your are an expert english grammarin and writer. Given a Multiple choice quiz for {subject} students. \
    You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use maximum of 50 for complexity.
    It the quiz is not at per with the cognitive and analytic abilities of the students,\
    update the quiz questions which needs to be changed and change the tone such that it perfectly fits the sudent's ability.
    Quiz_MCQs:
    {quiz}
    check from an expert English writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables = ["subject","quiz"],
    template = TEMPLATE2
)

review_chain =LLMChain(llm=llm,prompt=quiz_evaluation_prompt,output_key="review",verbose=True)

# made 2 chains now we will connect it via sequential chain
generate_chain = SequentialChain(chains=[quiz_chain,review_chain],input_variables=["text","number","subject","tone","response_json"],output_variables=["quiz","review"],verbose=True)