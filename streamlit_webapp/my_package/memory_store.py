from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import numpy as np
from my_package.prompts import SUMMARY_PROMPT
from my_package.logger import setup_colored_logging
import time
file_logger =setup_colored_logging(__name__)
# Initialize the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Replace with your preferred model


class SummaryResponse(BaseModel):
    message_summary :str = Field(description="This field contains summary for the chat message between Human and AI response.")
    rolling_summary :str =Field(description="This field contains summary of the previous conversations and updated with the new chat summary to create roling summary as whole.")
    long_term_memory:str|None = Field(description="This field contains import details about the user if conveyed in the conversation else None")

# Neo4j connection handler
class Neo4jHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def run_query(self, query, parameters=None):
        return self.driver.execute_query(query,parameters_=parameters)
    
    def transaction(self,query,parameters):
        with self.driver.session() as session:
            with session.begin_transaction() as tx:
                result = tx.run(query, parameters)
                return result



# Function to create a new message node dynamically
def update_memory(db:Neo4jHandler, llm, session_id, user_query, ai_response, message_id, current_summary):
    start_time = time.time()
    file_logger.info("running function")
    
    try:
        parser = PydanticOutputParser(pydantic_object=SummaryResponse)
        file_logger.info("Formating the prompt")
        prompt = PromptTemplate(
                                template=SUMMARY_PROMPT,
                                partial_variables={"format_instructions": parser.get_format_instructions()},
                            )
        message = prompt.format(query=user_query, response = ai_response, rolling_summary = current_summary)
        file_logger.info("Compiled Prompt...")
        file_logger .info(message)
    except Exception as e:
        file_logger.critical("There was some exception while configuing Prompts")
        file_logger.critical(e)
        
    file_logger.info("Trying to fetch structured output from the llm")
    try:
        for attempt in range(1,4):
            file_logger.info(f"Attempt {attempt} of getting the response from the LLM")
            response= llm.with_structured_output(SummaryResponse).invoke(message)
            file_logger.debug(f"Got this response from the LLM--->{response}")
            if response is not None:
                file_logger.info("Successfully got response from the LLM")
                break
            else:
                file_logger.warning("Got None as response")
                file_logger.warning("Retrying.....")
                continue
    except Exception as e:
        file_logger.critical("There was some error while getting response from the ")
        raise e
    try:
        message_summary = response.message_summary
        file_logger.info(f"The message summary is -->{message_summary}")
        rolling_summary = response.rolling_summary
        file_logger.info(f"The rolling summary is -->{rolling_summary}")
        long_term_memory = response.long_term_memory
        file_logger.warning(f"The long term memory is -->{long_term_memory}")
        file_logger.info("Generating model embeddings....")
        embedding_future = ThreadPoolExecutor().submit(embedding_model.encode, rolling_summary)
        file_logger.info("Embeddings generated +")
    
    except Exception as e:
        file_logger.critical(f"there was some while accessing LLM response ->{e}")
        
    file_logger.info("Inserting Data into Graph")
    query = """
    MATCH (s:Session {id: $session_id})
    CREATE (m:Message {message_id: $message_id, message_summary: $message_summary})
    WITH m, s  
    CALL db.create.setNodeVectorProperty(m, 'summary_embedding', $msummary_embedding)
    WITH m, s
    CREATE (s)-[:HAS_MESSAGE]->(m)
    SET s.summary = $rolling_summary
    RETURN m.message_id AS message_id
    """
    gst = time.time()
    try:
        message_embedding = embedding_future.result().tolist()
        result = db.run_query(query, parameters={
            "session_id": session_id,
            "message_id":message_id,
            "message_summary": message_summary,
            "msummary_embedding":message_embedding,
            "rolling_summary":rolling_summary,
        })
        file_logger.info("Successfully saved data")
        grt = time.time()- gst
        file_logger.info(result)
    except Exception as e:
        file_logger.critical("An error occured while inserting data into the graph")
        raise e
    
    file_logger.info("trying to create an index on the message and the message summary")
    # try:
    # # Step 6: Create index for the new message embeddings
    #     index_query = """CREATE VECTOR INDEX message_embeddings 
    #     FOR (m:Message) 
    #     ON m.summary_embedding
    #     OPTIONS {indexConfig: {
    #             `vector.dimensions`: 384,
    #             `vector.similarity_function`: 'cosine'}
    #         }"""
    #     db.run_query(index_query)
        
    # except Exception as e:
    #     file_logger.critical("There was some error while creating an index")
    #     file_logger.critical(e)
        
    
    file_logger.info(f"The function took -->{time.time()-start_time}")
    file_logger.info(f"The graph exceutio took -->{grt}")

    
    return result



#we have to edit this function--------------------
# Function to retrieve relevant messages using vector similarity
def retrieve_relevant_context(db, session_id, query_embedding, top_k=5):
    """
    Retrieve the most relevant context for a given search prompt using vector indexes in Neo4j.
    
    Args:
        driver: Neo4j driver instance.
        session_id (str): The session ID to filter messages.
        search_prompt (str): The prompt/query to search for relevant context.
        top_k (int): The number of top results to retrieve.
        openai_token (str): The OpenAI token for vector encoding.
        db_name (str): The Neo4j database name to use.
    
    Returns:
        list of tuples: Each tuple contains the session summary, message summary, and similarity score.
    """
    # Cypher query using vector encoding and querying the vector index
    
    query = """
    MATCH (s:Session {id: $session_id})-[:HAS_MESSAGE]->(m:Message)
    CALL db.index.vector.queryNodes('message_embeddings', $top_k, $query_embedding)
    YIELD node, score
    RETURN s.summary AS session_summary, node.message_summary AS message_summary, score
    """
    file_logger.info("Tring to run query...")
    try:
        result = db.run_query(query, {"session_id": session_id, "query_embedding": query_embedding, "top_k": top_k})
    except Exception as e:
        file_logger.critical("There was some problem while querying the Graph DB!")
        file_logger.critical(e)
        raise e
    return result


# Function to handle the entire memory process
def get_context_from_memory(db, session_id:str, user_query:str,top_k:int=5):
    srt = time.time()
    file_logger.info("Trying to encode user query...")
    try:
    # Step 1: Generate embedding for the current user query
        query_embedding = embedding_model.encode(user_query,show_progress_bar=True)
    except Exception as e:
        file_logger.error("There was some error while encoding user query")
        file_logger.critical(e)
        raise e
    # Step 2: Retrieve relevant messages from the graph
    file_logger.info("Trying to retrieve the relevant messages")
    try:
        relevant_messages = retrieve_relevant_context(db, session_id, query_embedding, top_k)
    except Exception as e:
        file_logger.error("There was some error while retreving relervant messages")
        file_logger.critical(e)
        raise e
    # state["chat_context"].append([relevant_messages])
    file_logger.info(f"The retrieved messages are --->{relevant_messages.records}")
    file_logger.info(f"The whole process took--{srt-time.time()}")
    return None


