from unittest.mock import MagicMock
import unittest
from my_package.memory_store import update_memory,Neo4jHandler,get_context_from_memory
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import os 
# from neo4j import GraphDatabase
from dotenv import load_dotenv
import logging
import sys
load_dotenv()


handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logging.getLogger("neo4j").addHandler(handler)
logger =logging.getLogger("neo4j").setLevel(logging.DEBUG)

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  
groq_api = os.getenv("GROQ_API_KEY")
if groq_api is None:
    raise ValueError("GROQ_API_KEY environment variable is not set or is empty!")

LLM_final = ChatGroq(api_key= groq_api, model="llama-3.1-70b-versatile")
neo4j_handler = Neo4jHandler(uri=uri,user=username,password=password)

class TestUpdateMemoryFunction(unittest.TestCase):
    def setUp(self):
        # Mock database connection
        self.mock_db = neo4j_handler

        # Mock LLM
        self.mock_llm = LLM_final


        # Mock embedding model
        self.mock_embedding_model = embedding_model

        # Inputs
        self.session_id = "sid_345"
        self.user_query = "That makes sense. You’re connecting this to the job listings app we discussed earlier?"
        self.ai_response = "Exactly! You mentioned it a while back—using user preferences as part of the search criteria. It seemed relevant to bring up here since indexing plays a crucial role in how quickly those searches return results."
        self.message_id = 3
        self.current_summary = "The user inquired about Neo4j indexing and sought the AI's expertise on implementing indexing techniques to optimize queries. The conversation progressed to compound indexes, which the AI recommended for efficiently searching job listings based on user preferences, specifically by title and location."

    def test_update_memory(self):
        
        result = update_memory(
            db=self.mock_db,
            llm=self.mock_llm,
            session_id=self.session_id,
            user_query=self.user_query,
            ai_response=self.ai_response,
            message_id=self.message_id,
            current_summary=self.current_summary
        )

        # Replace the actual embedding model and database with mocks
        # Call the function
        # result = get_context_from_memory(
        #     db=self.mock_db,
        #     session_id=self.session_id,
        #     user_query=self.user_query,
        #     top_k=2
        # )
        # print(result)
        # # Assertions
        # self.assertEqual(result, None)  # Assuming the query returns None if successful

if __name__ == "__main__":
    
    unittest.main()
    # print(uri)

    # try:
    #     driver = GraphDatabase.driver(uri, auth=(username, password))
    #     driver.verify_connectivity()
    #     print("Successfull")
    # except Exception as e:
    #     print(e)    