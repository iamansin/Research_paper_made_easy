import os
from uuid import uuid4
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain_huggingface  import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from my_package.chroma_manager import ChromaManager
from dotenv import load_dotenv
load_dotenv()

HF_Token = os.getenv("HUGGINGFACE_API")

hf_embeddings = HuggingFaceEndpointEmbeddings(
    model= "sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=HF_Token,
)

# class CustomEmbeddingFunction:
#     def __init__(self, api_endpoint):
#         self.api_endpoint = api_endpoint

#     def __call__(self, docs):
#         # Prepare the data for the API call
#         data = {"input": docs}

#         # Make the API call to the custom embedding model API
#         response = requests.post(self.api_endpoint, json=data)

#         # Handle the response from the custom embedding model API
#         if response.status_code == 200:
#             embeddings = response.json()["embeddings"]  # Extract the embeddings from the response
#             return embeddings
#         else:
#             print("Error:", response.status_code)
#             return None
              
def PdfToText(file_names,pdf_files):
    doc = []
    i=0
    for pdf in pdf_files:
        file_name = file_names[i]
        i+=1
        pdf_reader = PdfReader(pdf)
        try:
            for page_number, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                document = Document(page_content=text, metadata={"source": file_name, "page": page_number})
                doc.extend([document])
        except Exception as e:
            print(f"an error occoured at line 60 DBChroma :{e}")
    return doc
            
def PdfToChunks(file_names,pdf_files):
    documents = PdfToText(file_names,pdf_files)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000,chunk_overlap=200)
    try :
        chunked_documents = text_splitter.split_documents(documents)
    except Exception as e:
        print (f"an error occured at line 69 DBChroma :{e}")
    return chunked_documents


def Convert_pdf_to_embedding(file_names,pdf_files):
    chunked_documents = PdfToChunks(file_names,pdf_files)
    document_ids = [str(uuid4()) for _ in range(len(chunked_documents))]    
    Chroma_Manager = ChromaManager(persist_directory="chroma_database", collection_name="research_papers")
    try:
        Chroma_Manager.add_documents(chunked_documents,hf_embeddings,document_ids) 
        Chroma_Manager.persist()
        Chroma_Manager.close()
        print("Documents converted and embeddings created successfully.")
    except Exception as e:
        print(f"Error at line 77 DBChroma {e}")
    
    

    


