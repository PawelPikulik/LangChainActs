import os
import openai
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

question = input("Ask a question: ").strip()

reasoning_prompt = "{question}\nExplain your answer in key steps."

cot_prompt = ChatPromptTemplate(
    messages=[("human", reasoning_prompt)],
    input_variables=["question"]
)
llm = ChatOpenAI(model="gpt-5.2")

llm_chain = cot_prompt | llm | StrOutputParser()
print(f"\n-- Chain of Thought --\n")
print(llm_chain.invoke(question))