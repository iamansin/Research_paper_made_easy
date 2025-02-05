import os
import logging
from uuid import uuid4
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
from dotenv import load_dotenv
from typing import List,Union
from my_package.chroma_manager import ChromaManager
from my_package.documentDB_operations import add_file_and_documents
load_dotenv()

HF_Token = os.getenv("HUGGINGFACE_API")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def insert_into_chroma(chunked_docs: List[Document], doc_ids: List[str]) -> Union[str, Exception]:
    """
    Inserts documents into ChromaDB.

    Args:
        chunked_docs (List[dict]): List of document chunks to be inserted.
        doc_ids (List[str]): List of document IDs corresponding to the chunks.

    Returns:
        Union[str, Exception]: Success message or exception if an error occurs.
    """
    if len(chunked_docs) != len(doc_ids):
        logger.error("Mismatch: The number of documents does not match the number of IDs.")
        raise("Document and ID count mismatch.")

    # Initialize ChromaManager
    logger.info("Initializing Chroma Manager...")
    manager = ChromaManager(
        host="localhost",
        port=8000,
        collection_name="my_collection",
        model_name="all-MiniLM-L6-v2"
    )

    logger.info("Connecting to ChromaDB server...")
    try:
        # Connect to ChromaDB
        response = await manager.connect()
        if isinstance(response,Exception):
            logger.error("Failed to connect to ChromaDB server.")
            raise ("Failed to connect to ChromaDB server.")
        # Add documents to ChromaDB
        logger.info("Adding documents to ChromaDB...")
        response = await manager.add_documents(documents=chunked_docs, ids=doc_ids)
        if isinstance(response,Exception):
            logger.error("Failed to add Documents")
            raise ("Failed to add Documents to ChromaDB")
        logger.info("Documents successfully inserted into ChromaDB.")
        return "Insertion successful."

    except ConnectionError as ce:
        logger.exception("Connection error occurred while connecting to ChromaDB.")
        raise ce
    except Exception as e:
        logger.exception("An unexpected error occurred.")
        raise e

def PdfToText(file_names,pdf_files):
    doc = []
    for i, pdf in enumerate(pdf_files):
        file_name = file_names[i]
        pdf_reader = PdfReader(pdf)
        try:
            file_documents = []
            for page_number, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                document = Document(page_content=text, metadata={"source": file_name, "page": page_number})
                file_documents.extend([document])
            doc.append(file_documents)
        except Exception as e:
            logger.error(f"an error occoured at line 60 DBChroma :{e}")
            
    return doc
            
def PdfToChunks(file_names,pdf_files):
    documents = PdfToText(file_names,pdf_files)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000,chunk_overlap=200)
    try :
        chunked_documents = []
        for i, document in enumerate(documents):
            file_id = str(uuid4())
            chunked_document = text_splitter.split_documents(document)
            logger.warning(f"The length of chunked document is {len(chunked_document)} and type is {type(chunked_document)}")
            document_ids = [str(uuid4()) for _ in range(len(chunked_document))]  
            document_dict = {"file_name": file_names[i],"file_id":file_id,"file_document_ids":document_ids,"file_documents":chunked_document}
            # logger.info(f"The chunked dict is {document_dict}")
            chunked_documents.append(document_dict)
            
        return chunked_documents
    except Exception as e:
        logger.error(f"an error occured at line 69 DBChroma :{e}")
        raise e
    


def Convert_pdf_to_embedding(file_names,pdf_files):
    logger.info(f"The file names are:{file_names}")
    try:
        chunked_documents_with_ids= PdfToChunks(file_names,pdf_files)
    except Exception as e:
        logger.critical(f"An error occured while creating chunks of the file {e}")
        raise e
    try:
        logger.info("Now saving files to SQLite")
        add_file_and_documents(chunked_documents_with_ids)
        logger.info("Successfully saved files in ")
    except Exception as e:
        logger.critical(f"An error occured while saving files to the SQLite {e}")
        raise (f"There was some error{e}")
    chunked_documents = [document for doc in chunked_documents_with_ids for document in doc["file_documents"]]
    document_ids =[doc_id for docs in chunked_documents_with_ids for doc_id in docs["file_document_ids"]]
    logger.info(f"length of chunked_documents {len(chunked_documents)}, length of ids {len(document_ids)}")
    try:
        logger.info("Saving files in ChromaDB")
        asyncio.run(insert_into_chroma(chunked_docs=chunked_documents, doc_ids=document_ids))
        logger.info("Documents converted and embeddings created successfully.")
    except Exception as e:
        logger.critical(f"Error occured while inserting documents to chromadb{e}")
        raise e
    
    

    


