from sqlalchemy import create_engine, Column, String, Integer, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import desc
import uuid

# SQLAlchemy setup
Base = declarative_base()


# ChatSession Table
class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String)

# ChatMessages Table
class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('chat_sessions.id'))
    message = Column(Text)
    sender = Column(String)

# Database setup
engine = create_engine('sqlite:///chat_app.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Functions for CRUD operations
def get_all_sessions():
    """Retrieve all chat sessions."""
    db_session = Session()
    sessions = db_session.query(ChatSession).all()
    db_session.close()
    return sessions

def create_session(session_name):
    """Create a new chat session."""
    db_session = Session()
    new_session = ChatSession(name=session_name)
    db_session.add(new_session)
    db_session.commit()
    db_session.refresh(new_session)  # Refresh to access the auto-generated ID
    db_session.close()
    return new_session


def get_chat_messages(session_id):
    """Retrieve all messages for a given session."""
    db_session = Session()
    messages = db_session.query(ChatMessage).filter(ChatMessage.session_id == session_id).all()
    db_session.close()
    return messages

def delete_session(session_id):
    """Delete a chat session and associated messages."""
    db_session = Session()
    try:
        db_session.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
        db_session.query(ChatSession).filter(ChatSession.id == session_id).delete()
        db_session.commit()
    except Exception as e:
        print(f"Error deleting session: {e}")
        db_session.rollback()
    finally:
        db_session.close()

def store_chat_message(session_id, role, content):
    """
    Stores a chat message in the database.

    Args:
        session_id (str): The ID of the chat session.
        role (str): The role of the sender ("user" or "assistant").
        content (str): The message content.
    """
    db_session = Session()
    try:
        # Create a new ChatMessage instance
        new_message = ChatMessage(
            session_id=session_id,
            sender=role,
            message=content
        )
        # Add and commit the message to the database
        db_session.add(new_message)
        db_session.commit()
    except Exception as e:
        print(f"Error storing chat message: {e}")
        db_session.rollback()
    finally:
        db_session.close()
        

def get_chat_messages_by_session(session_id: str):
    """
    Fetch all chat messages for a given session ID.
    Args:
        db_session (Session): The database session.
        session_id (str): The session ID to filter messages by.

    Returns:
        list: A list of dictionaries containing chat messages.
    """
    try:
        db_session = Session()
        messages = (
            db_session.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.asc())  # Sort by message ID (or timestamp if applicable)
            .all()
        )
        if not messages:
                return []

        # Transform messages to the desired format
        return [{"role": message.sender, "content": message.message} for message in messages]

    except Exception as e:
        # Handle any unexpected errors during the query
        print(f"An error occurred while fetching chat messages: {str(e)}")
        return []
    
