from chromadb.utils import embedding_functions
from chromadb import AsyncHttpClient
from typing import List,Union
import numpy as np
from langchain.docstore.document import Document
import logging
import numpy as np


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ChromaManager:
    def __init__(self, host: str, port: int, collection_name: str, model_name: str | None):
        """
        Initialize the ChromaManager with connection details and embedding model.
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name) if model_name else None
        self.client = None
        self.collection = None

    async def connect(self) -> bool:
        """
        Connect to the ChromaDB server and initialize the client.
        """
        self.client = await AsyncHttpClient(host=self.host, port=self.port)
        try:
            # Try loading the existing collection
            self.collection = await self.client.get_collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load collection '{self.collection_name}': {str(e)}")
            if "does not exist" in str(e).lower():
                try:
                    logger.info(f"Collection '{self.collection_name}' not found. Creating a new one.")
                    self.collection = await self.client.create_collection(
                        name=self.collection_name, embedding_function=self.embedding_fn
                    )
                    logger.info(f"Collection '{self.collection_name}' created successfully.")
                    return True
                except Exception as e:
                    logger.error("An error occurred while creating a new collection.", exc_info=True)
                    raise e
            else:
                logger.error("An unexpected error occurred.", exc_info=True)
                raise e

    async def add_documents(self, documents: List[Document], ids: list) -> bool:
        """
        Add documents to the collection.

        Args:
            documents (List[Document]): List of Document objects to add.
            ids (list): List of unique document IDs corresponding to the documents.

        Returns:
            bool: True if documents are added successfully.
        """
        if not self.collection:
            logger.error("Collection not initialized. Call 'connect' first.")
            raise Exception("Collection not initialized. Call 'connect' first.")
        try:
            document_text = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            await self.collection.add(
                documents=document_text,
                metadatas=metadatas,
                ids=ids
            )
            logger.info("Documents added successfully.")
            return True
        except Exception as e:
            logger.error("Error adding documents.", exc_info=True)
            raise e

    async def query(self, query: Union[List[float], np.ndarray], n_results: int = 5) -> dict:
        """
        Perform similarity search on the collection.

        Args:
            query (Union[List[float], np.ndarray]): Query embeddings for similarity retrieval.
            n_results (int): Number of similar documents to retrieve.

        Returns:
            dict: Retrieved documents and their metadata.
        """
        if not self.collection:
            logger.error("Collection not initialized. Call 'connect' first.")
            raise Exception("Collection not initialized. Call 'connect' first.")
        try:
            results = await self.collection.query(query_embeddings=query, n_results=n_results)
            logger.info(f"Query executed successfully. Retrieved {len(results['documents'])} results.")
            return results
        except Exception as e:
            logger.error("Error querying collection.", exc_info=True)
            raise e

    async def delete_data(self, ids: list) -> None:
        """
        Delete data from the collection.

        Args:
            ids (list): List of IDs of the documents to delete.
        """
        if not self.collection:
            logger.error("Collection not initialized. Call 'connect' first.")
            raise Exception("Collection not initialized. Call 'connect' first.")
        try:
            await self.collection.delete(ids=ids)
            logger.info(f"Successfully deleted documents with IDs: {ids}")
        except Exception as e:
            logger.error("Error occurred while deleting files.", exc_info=True)
            raise e
