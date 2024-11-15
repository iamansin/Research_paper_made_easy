import streamlit as st 
from my_package.doc_embedding import Convert_pdf_to_embedding
from my_package.mongo_manager import MongoManager
from my_package.sqlite_manager import SQLiteManager, insert_pdf_to_sqlite
import time

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

@st.cache_resource
def Create_embeddings(document_dict,files):
    file_names =[]
    for file in document_dict:
        file_names.append(file['filename'])
    try:
        Convert_pdf_to_embedding(file_names,files)
        st.success("Embeddings generated and stored in Chroma!")
        return "successful"
    except Exception as e:
        print(f"error occured as :{e}")
        return None
            

def insert_files_to_db(document_dict):
    if document_dict:
        try:
            # creating instance of Mongo_DB class...
            mongo_db = MongoManager()
            #Storing pdfs in mongodb and then returning inserted ids..
            doc_ids = mongo_db.insert_documents(document_dict)
            mongo_db.close_connection()
        except Exception as e:
            print(f"Error at line 27(submit docs) {e}")
            return None
        
        try:
            # creating an SQLite_DB object 
            sqlite_db = SQLiteManager()
            # Store the PDF names with their corresponding MongoDB IDs in SQLite
            for mongo_id, file in zip(doc_ids, document_dict):
                filename = file['filename'] 
                insert_pdf_to_sqlite(sqlite_db,str(mongo_id), filename)
            sqlite_db.close_connection() 
            st.success(f"Uploaded {len(doc_ids)} files successfully!")
            return "Successfull"
        except Exception as e:
            print("Failed to add file to sqlite db")
            return None
        

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

def main():
    
    st.title("Upload Your Papers Here!")
    
    # Upload button
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        # Store uploaded files in session state
        st.session_state.uploaded_files = uploaded_files
        
    # Check if files have been uploaded and stored in session state
    if st.session_state.uploaded_files:
        submit_button = st.button("Submit")
        # Only process files when the user clicks submit
        if submit_button:
            document_dict = []
            for file in st.session_state.uploaded_files:
                file_data = file.read()
                document = {
                    "filename": file.name,
                    "content": file_data
                }
                document_dict.append(document)
            
            # Saving data into MongoDB
            with st.spinner("Saving Your PDF Files!"):
                time.sleep(1)
                response = insert_files_to_db(document_dict)
                if response is not None :
                    st.success("Files successfully saved to the database!")
                else:
                    st.error(f"There was some error storing your data")
            
            # Creating embeddings and saving them 
            with st.spinner("Generating Embeddings!"):
                response = Create_embeddings(document_dict, st.session_state.uploaded_files)
                if response:
                    st.write("We Are Ready for Help. Please go to Knowlege-page for information retrieval !")
                else:
                    st.error(f"There was some error generating embeddings")
    else:
        st.info("Please upload PDF files to proceed.")

main()