from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()  # Load variables from .env file

api_key = os.getenv(OPENAI_API_KEY)

# Create the model object
llm = ChatOpenAI(
    model="gpt-4o-audio-preview",  # Specifying the model
    temperature=0,  # Controls randomness in the output
    max_tokens=None,  # Unlimited tokens in output (or specify a max if needed)
    timeout=None,  # Optional: Set a timeout for requests
    max_retries=2  # Number of retries for failed requests
)
