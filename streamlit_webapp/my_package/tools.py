from langchain_community.tools import tool
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
import os 
from dotenv import load_dotenv
from chroma_manager import ChromaManager
load_dotenv()
HF_Token = os.getenv("HUGGINGFACE_API")

hf_embeddings = HuggingFaceEndpointEmbeddings(
    model= "sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=HF_Token,
)
#name="vector_search", description="Searches the vector store for relevant information in submitted research papers."
@tool
def search_vector_store(query: str):
    """This is a Tool used for searching relevant documents in submitted research papers."""
    Chroma_Manager = ChromaManager(persist_directory="chroma_database", collection_name="research_papers")
    try:
        retriever = Chroma_Manager.retriever(hf_embeddings)
        vector_results = retriever.query(query_text=query, n_results=5)
    except Exception as e:
        print(f"An Error Occured while quering Chroma database::{e}")
    return vector_results

#name="web_search", description="Searches the web for simpler explanations of complex concepts."
@tool
def search_web(query: str):
    """Tool to perform a web search for simpler explanations using DuckDuckGo."""
    search_api_wrapper = DuckDuckGoSearchAPIWrapper()
     # Initialize DuckDuckGo search wrapper
    duckduckgo_search = DuckDuckGoSearchRun(api_wrapper=search_api_wrapper)
    try:
        web_results = duckduckgo_search.invoke(query)
        return web_results
    except Exception as e:
        print(f"There occured an Error with DuckDuckGO tool ::{e}")
        return "Not able to get results"
    

#name="arxiv_search", description="Finds similar research papers from arXiv based on the query."
@tool
def search_arxiv(query: str):
    """This a Tool used for finding similar research paper by quering arxiv."""
    arxiv = ArxivAPIWrapper(
        top_k_results = 3,
        ARXIV_MAX_QUERY_LENGTH = 300,
        load_max_docs = 3,
        load_all_available_meta = False,
        doc_content_chars_max = 40000
        )
    try:
        arxiv_results = arxiv.run(query)  # placeholder for arXiv search logic
    except Exception as e :
        print(f"there is an Error occured while quering arxiv::{e}")
    return arxiv_results

# Now we simply list the tools for use in the agent setup
TOOLS = [search_vector_store, search_web, search_arxiv]
