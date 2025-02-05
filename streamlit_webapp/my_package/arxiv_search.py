from typing_extensions import Optional,List,Dict
import asyncio
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from sentence_transformers import SentenceTransformer, util 
from my_package.logger import setup_colored_logging
logger = setup_colored_logging(name=__name__)



async def search_arxiv(query: str) -> Optional[List[Dict[str, str]]]:
    """
    Search arXiv for research papers most relevant to the query.

    Args:
        query (str): The search query.

    Returns:
        Optional[List[Dict[str, str]]]: A list of dictionaries containing paper details sorted by relevance.
    """
    # Initialize arXiv API wrapper
    logger.info(f"The query is -->{query}")
    arxiv = ArxivAPIWrapper(
        top_k_results=5,
        ARXIV_MAX_QUERY_LENGTH=300,
        load_max_docs=5,
        load_all_available_meta=True,
        doc_content_chars_max=40000,
    )
    
    try:
        logger.info("Fetching results from arXiv...")
        # Retrieve summaries as documents
        arxiv_results = await asyncio.to_thread(arxiv.get_summaries_as_docs, query)
        if not isinstance(arxiv_results, list):
            logger.warning("Unexpected response format from arXiv API.")
            return None
        
        # Extract and structure paper details
        structured_results = [
            {
                "Title": paper.metadata.get("Title", "Unknown Title"),
                "Summary": paper.page_content or "No summary available.",
                "Published": paper.metadata.get("Published", "Unknown Date"),
                "Link": paper.metadata.get("Entry ID", "No link available"),
            }
            for paper in arxiv_results
        ]
        logger.info("Successfully fetched and structured results from arXiv.")
    except Exception as e:
        logger.critical(f"Error occurred while querying arXiv: {e}")
        return None
    
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and effective
    # Generate query embedding
    try:
        logger.info("Generating query embedding...")
        query_embedding = await asyncio.to_thread(model.encode, query, convert_to_tensor=True)
        logger.info("Query embedding generated successfully.")
    except Exception as e:
        logger.critical(f"Error generating query embedding: {e}")
        return None

    # Score papers by cosine similarity
    scored_results = []
    for result in structured_results:
        abstract = result["Summary"]
        if not abstract:
            logger.warning(f"No abstract found for: {result['Title']}")
            continue
        try:
            abstract_embedding = await asyncio.to_thread(model.encode,abstract, convert_to_tensor=True)
            similarity = util.cos_sim(query_embedding, abstract_embedding).item()
            scored_results.append((similarity, result))
        except Exception as e:
            logger.error(f"Error encoding abstract for '{result['Title']}': {e}")
            continue

    # Sort results by similarity score
    scored_results.sort(key=lambda x: x[0], reverse=True)

    # Format and return the top results
    top_results = [
        {
            "Title": res[1]["Title"],
            "Link": res[1]["Link"],
            "Summary": res[1]["Summary"][:100],  # Trim long summariess
            "Relevance Score": round(res[0], 4),
        }
        for res in scored_results
    ]
    logger.info("Top results formatted and ready for return.")
    # logger.info(top_results)
    return top_results
