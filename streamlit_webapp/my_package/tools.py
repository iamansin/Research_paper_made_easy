from langchain_community.tools import tool
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.utilities import GoogleSerperAPIWrapper
from duckduckgo_api_haystack import DuckduckgoApiWebSearch
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
import numpy as np
import os 
import asyncio
from typing import List,Optional,Union
from dotenv import load_dotenv
from my_package.chroma_manager import ChromaManager
from my_package.logger import setup_colored_logging

load_dotenv()
HF_Token = os.getenv("HUGGINGFACE_API")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

hf_embeddings = HuggingFaceEndpointEmbeddings(
    model= "sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=HF_Token,
)

logger = setup_colored_logging(name=__name__)



async def query_chroma(query_embeddings: Union[List[float], np.ndarray]) -> Union[List[dict], str]:
    """
    Queries ChromaDB to retrieve the most relevant results for the given embeddings.

    Args:
        query_embeddings (List[float]): The query embeddings to search for.

    Returns:
        Union[List[dict], str]: A list of search results or an error message.
    """
    if isinstance(query_embeddings, np.ndarray):
        # Convert NumPy array to list for compatibility
        logger.info("The current query is encoded to numpy array")

    
    # Initialize ChromaManager
    manager = ChromaManager(
        host="localhost",
        port=8000,
        collection_name="my_collection",
        model_name="all-MiniLM-L6-v2",
    )

    try:
        # Connect to ChromaDB
        response = await manager.connect()
        if not response:
            logger.error("Failed to connect to ChromaDB.")
            return "Failed to connect to ChromaDB."
        
        # Perform the query
        results = await manager.query(query=query_embeddings, n_results=5)
        return results

    except Exception as e:
        logger.exception("An error occurred while querying ChromaDB.")
        raise e

@tool
def search_vector_store(query: str) -> Optional[List[dict]]:
    """
    Searches a vector store for documents relevant to the given query.

    Args:
        query (str): The query string to search for.

    Returns:
        Optional[List[dict]]: A list of relevant documents or None if an error occurs.
    """
    if not query:
        logger.error("Query string cannot be empty.")
        return None

    try:
        # Encode the query using SentenceTransformer
        logger.info("Encoding query...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        encodings = model.encode(query)
        logger.info("Query encoded successfully.")

        # Query ChromaDB
        logger.info("Querying ChromaDB...")
        results = asyncio.run(query_chroma(encodings))
        if results:
            num_docs = len(results["ids"][0])
            logger.info("Results retrieved successfully from ChromaDB.")
            
            # Extract and organize the results
            scored_results = [
                (
                    results.get("distances")[0][i],  # Distance
                    results.get("documents")[0][i],  # Document content
                    results.get("metadatas")[0][i]   # Metadata
                )
                for i in range(num_docs)
        ]
            logger.info("Results retrieved successfully from ChromaDB.")
        else:
            logger.warning("No results found.")
        return scored_results

    except Exception as e:
        logger.exception("An error occurred while searching the vector store.")
        return None



#Logic for Web Search Tool......
def google_serper(query: str):
    """Search using Google Serper API."""
    search_module = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
    logger.info("Searching query on Google Serper...")
    response = search_module.run(query)
    logger.info("Successfully got results from Google Serper...")
    return response


def duck_duck_go(query: str):
    """Search using DuckDuckGo."""
    try:
        logger.info("Trying the first DDG search method....")
        query = query.strip()[:200]  # Sanitize and truncate query
        response = DDGS().text(
            keywords=query,
            region="wt-wt",
            safesearch="off",
            max_results=50
        )
        logger.info("Successfully got results from DDG search method")
        return response
    except Exception as e:
        try:
            logger.error(f"First method failed: {e}")
            logger.info("Trying the second DDG search method....")
            websearch = DuckduckgoApiWebSearch(
                top_k=10,
                max_results=10,
                region="wt-wt",
                safesearch="moderate",
                timelimit=None,
                backend="api",
                timeout=10,
                use_answers=False
            )
            response = websearch.run(query)
            logger.info("Successfully got results from second DDG search method")
            return response
        except Exception as e:
            logger.error(f"Got error from DDG second method")
            raise e

def run_web_query(query: str):
    """Web search logic combining Google Serper and DuckDuckGo."""
    try:
        logger.info("Trying Google Serper...")
        res = google_serper(query)
        return res
    except Exception as e:
        logger.warning("Google Serper failed")
        logger.error(f"Got error from Google Serper{e}")
    try:
        logger.info("Now Trying DuckDuckGo Search...")
        res = duck_duck_go(query)
        return res
    except Exception as e:
        logger.warning("DuckDuckGo failed")
        logger.error(f"Got error from DuckDuckGO{e}")
        raise e


@tool
def search_web(query: str) -> str:
    """
    Tool to perform a web search using Google Serper and DuckDuckGo.
    Falls back to DuckDuckGo if Google Serper fails.
    Returns a plain text summary of results.
    """
    try:
        logger.info("Running search_web Tool...")
        results = run_web_query(query)
        logger.info("Successfully got resuts from search tool")
        return results
    except Exception as e:
        logger.warning("After trying multiple tools still got no response!")
        logger.error(f"Got error: {e}")
        raise e


# Now we simply list the tools for use in the agent setup
TOOLS = [search_vector_store, search_web]
