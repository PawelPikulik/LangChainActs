import os

import torch
from transformers import pipeline
from transformers.utils import logging as hf_logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

customer_email = {"email": "Does daily coffee perfection, leave you emotionally confident?"}

template = """
Given the text, decide what is the issue the customer is concerned about. Valid categories are these:
* product issues
* delivery problems
* missing or late orders
* wrong product
* cancellation request
* refund or exchange
* bad support experience
* no clear reason to be upset

Text: {email}
Category:
"""

# Keep HF pipeline quiet
hf_logging.set_verbosity_error()

sentiment_model = pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
)

raw_sentiment = sentiment_model(customer_email["email"])[0]  # {'label': 'LABEL_1', 'score': ...}

# cardiffnlp/twitter-roberta-base-sentiment uses numeric labels.
label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
sentiment_result = label_map.get(raw_sentiment.get("label"), raw_sentiment.get("label"))
sentiment_score = raw_sentiment.get("score")

inputs = {
    "email": customer_email["email"],
    "sentiment_result": sentiment_result,
}

prompt = PromptTemplate(template=template, input_variables=["email", "sentiment_result"])

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001",)

chain = prompt | llm | StrOutputParser()

category = chain.invoke(inputs).strip()

if isinstance(sentiment_score, (int, float)):
    print(f"Email: {customer_email['email']}")
    print(f"Category: {category}")
    print(f"Sentiment analysis: {sentiment_result}")

