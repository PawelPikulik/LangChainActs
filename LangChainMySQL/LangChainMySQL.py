import os
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_classic.chains import ConversationChain

def get_chat_history(session_id: str, db_path: str = "chat_memory.db"):
    """
    Auto-detects database:
    1. Uses MySQL if DB_URL env var is set and MySQL is reachable
    2. Falls back to SQLite (zero-config) otherwise
    """
    
    # Check for MySQL connection string
    mysql_url = os.getenv("DB_URL")
    
    if mysql_url and _mysql_available(mysql_url):
        print(f"Using MySQL for session: {session_id}")
        return SQLChatMessageHistory(
            session_id=session_id,
            connection_string=mysql_url
        )
    
    # Fallback to SQLite - no server needed!
    sqlite_url = f"sqlite:///{db_path}"
    print(f"Using SQLite for session: {session_id}")
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=sqlite_url
    )

def _mysql_available(url: str) -> bool:
    """Test if MySQL is actually reachable"""
    try:
        from sqlalchemy import create_engine
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False
        
def main():
    # For portable app, put DB next to executable
    db_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(db_dir, "chat_history.db")

    # Zero-config: works out of the box
    history = get_chat_history(session_id="user_123", db_path=db_path)
    memory = ConversationBufferMemory(chat_memory=history)

    llm = ChatOpenAI(model="gpt-4")
    chain = ConversationChain(llm=llm, memory=memory)

    # First run → creates SQLite file automatically
    chain.predict(input="Hello!")

if __name__ == "__main__":
    main()
