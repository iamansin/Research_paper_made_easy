
from langchain_community.vectorstores import Chroma  # Ensure using LangChain's Chroma

class ChromaManager:
    def __init__(self, persist_directory: str, collection_name: str):
        """Initialize the ChromaManager with persistence and collection details."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.db = None  # The main database instance

    def add_documents(self, documents, embedding_func, doc_ids=None):
        try:
            # Add documents using LangChain's Chroma interface
            self.db = Chroma.from_documents(
                documents=documents,
                embedding=embedding_func,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                ids=doc_ids
            )
            print("Documents successfully added to Chroma.")
        except Exception as e:
            print(f"Error adding documents: {e}")
            return e

    def similarity_search(self, query_embedding, k=4):
        try:
            results = self.db.similarity_search(query_embedding, k)
            return results
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return None

    def similarity_search_with_scores(self, query_embedding, k=4):
        try:
            results = self.db.similarity_search_with_relevance_scores(query_embedding, k)
            return results
        except Exception as e:
            print(f"Error during similarity search with scores: {e}")
            return None

    def persist(self):
        try:
            self.db.persist()
            print("Collection persisted to disk.")
        except Exception as e:
            print(f"Error persisting collection: {e}")
            return e

    def retriever(self,embedding_func):
        try:
            self.db = Chroma(persist_directory=self.persist_directory,embedding_function=embedding_func)
            # Use LangChain's retrieval mechanism with the existing collection
            retriever = self.db.as_retriever()
            print("Retriever successfully initialized.")
            return retriever
        except Exception as e:
            print(f"Error initializing retriever: {e}")
            return None

    def close(self):
        try:
            self.db = None  # Clear the reference to the database
            print(f"Chroma database connection closed.")
        except Exception as e:
            print(f"Error closing the database connection: {e}")



