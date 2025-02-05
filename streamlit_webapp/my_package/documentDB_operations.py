from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.orm import relationship, declarative_base, sessionmaker
from sqlalchemy import create_engine
import asyncio
from my_package.chroma_manager import ChromaManager

Base = declarative_base()

class File(Base):
    __tablename__ = "files"
    id = Column(String, primary_key=True)  # File ID, provided explicitly
    file_name = Column(String, nullable=False)  # File name
    documents = relationship("Documents", back_populates="file", cascade="all, delete-orphan")

class Documents(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)  # Document ID, provided explicitly
    file_id = Column(String, ForeignKey("files.id"), nullable=False)  # Foreign key to File table
    file = relationship("File", back_populates="documents")
    
    
engine = create_engine("sqlite:///documents.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


def add_file_and_documents(files_dict):
    """
    Add a file and its associated documents to the database.
    
    :param files_dict: A List of dictonaries which contains data related to files
    """
    try:
        session = Session()
    
        for file_data in files_dict:
        # Create File instance
            file_id = file_data["file_id"]
            file_name = file_data["file_name"]
            new_file = File(id=file_id, file_name=file_name)
            session.add(new_file)
            
            documents = file_data["file_document_ids"]
            # Add associated documents
            for doc_id in documents:
                new_document = Documents(id=doc_id,file=new_file)
                session.add(new_document)
            
            session.commit()
            print("File and documents added successfully.")
    except Exception as e:
        session.rollback()
        print(f"Error while adding file and documents: {e}")


async def adelete_file(file_id_list):
    """
    Delete a file and its associated documents.
    
    :param file_id_list: List of ID/IDs of the file/files to delete
    """
    try:
        session = Session()
        chroma_db = ChromaManager(
        host="localhost",
        port=8000,
        collection_name="my_collection",
        model_name="all-MiniLM-L6-v2")
        
        try:
            for file_id in file_id_list:
                file_to_delete = session.query(File).filter(File.id == file_id).one_or_none()
                if file_to_delete:
                    document_ids = [doc.id for doc in session.query(Documents).filter(Documents.file_id == file_id).all()]
                    try:
                        response = await chroma_db.connect()
                        if isinstance(response,Exception):
                            # .error("Failed to connect to ChromaDB server.")
                            raise ("Failed to connect to ChromaDB server.")
                        # Add documents to ChromaDB
                        # logger.info("Adding documents to ChromaDB...")
                        await chroma_db.delete_data(ids=document_ids)
                    except Exception as e:
                        raise e
                    session.delete(file_to_delete)
                    session.commit()
                    print(f"File with ID {file_id} and its documents deleted successfully.")
                else:
                    print(f"No file found with ID {file_id}.")
        except Exception as e:
            session.rollback()
            print(f"Error while deleting file: {e}")
    except Exception as e:
        print(f"There was some error :{e}")
    
def delete_file(file_id_list):
    try:
        asyncio.run(adelete_file(file_id_list))
    except Exception as e:
        raise e

def get_all_files():
    """
    Retrieve all files and their document counts.
    
    :param session: SQLAlchemy session object
    :return: List of file details with document counts
    """
    try:
        session = Session()
        files = session.query(File).all()
        result = [
            {
                "file_id": file.id,
                "file_name": file.file_name,
            }
            for file in files
        ]
        return result
    except Exception as e:
        print(f"Error while retrieving files: {e}")
        return []
