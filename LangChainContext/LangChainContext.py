from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0
    max_tokens=300,
)

base_chain = prompt | llm | StrOutputParser()

history = InMemoryChatMessageHistory()
chain = RunnableWithMessageHistory(
    base_chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history",
)

session = {"configurable": {"session_id": "default"}}

print("Chat started. Type your message and press Enter.")
print("Type 'exit' or 'quit' to stop. Press Ctrl+C to force quit.\n")

try:
    while True:
        user_input = input("Ask: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            break

        response = chain.invoke({"input": user_input}, config=session)
        print(f"AI: {response}\n")
except KeyboardInterrupt:
    print("\nExiting...\n")
