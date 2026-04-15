import os
import sys
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv()

@tool
def primes_upto(n: int) -> str:
    """Return all prime numbers up to and including n."""

    def is_prime(x: int) -> bool:
        if x < 2:
            return False
        if x in (2, 3):
            return True
        if x % 2 == 0:
            return False
        i = 3
        while i * i <= x:
            if x % i == 0:
                return False
            i += 2
        return True

    primes = [str(i) for i in range(2, n + 1) if is_prime(i)]
    return ", ".join(primes)


llm = ChatOpenAI(model="gpt-5.2")
agent = create_agent(llm, tools=[primes_upto])

user_input = input("Enter prime number: ").strip()

result = agent.invoke(
    {"messages": [
        ("user", user_input)]
    })
print(result["messages"][-1].content)
