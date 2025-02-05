import streamlit as st 
from my_package.database_operations import get_all_sessions
from my_package.doc_embedding import Convert_pdf_to_embedding
from my_package.documentDB_operations import get_all_files,delete_file

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def Create_embeddings(document_dict,files):
    file_names =[file['filename'] for file in document_dict]
    try:
        Convert_pdf_to_embedding(file_names,files)
        st.success("Embeddings generated and stored in Chroma!")
        return "successful"
    except Exception as e:
        print(f"error occured as :{e}")
        return None
        

def main():
    icon_add = [":heavy_plus_sign:"]
    # icon_delete = [":material/delete:"]

    add_new_chat = st.sidebar.pills("`New Chat`", options=icon_add, selection_mode="single", default=None)

    if add_new_chat:
        st.sidebar.info("🔐 Open in Knowledge-Sruf")

    # Display saved sessions with delete buttons
    sessions = get_all_sessions()
    
        
    if sessions:
        st.sidebar.markdown("*Saved Chats* :speech_balloon:")
        # print(f"This is currently using session id for THE session--------->{st.session_state["selected_session"]}")
        for session in sessions[::-1]:
            col1, col2 = st.sidebar.columns([15, 1])
            with col1:
                if st.button(f"🌐 {session.name[:26].strip()}..", use_container_width=True, key=f"session_{session.id}",):
                    # print(f"This is current session id of SELECTED session--------->{st.session_state["selected_session"]}, session name :{session.name}")
                    st.info("🔐 Open in Knowledge-Sruf")
                    
    tab1, tab2 = st.tabs(["📥 Submit Documents", "📂 Saved Documents"])
    with tab1:
        st.header(":rainbow[Upload Your Papers Here!]")
        
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
            
        if "submit_status" not in st.session_state:
            st.session_state["submit_status"] =None
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
            
                # Creating embeddings and saving them 
                with st.spinner("Saving Your Files!"):
                    try:
                        Create_embeddings(document_dict, st.session_state.uploaded_files)
                        st.success("Successfully saved you Documents ")
                        st.write("We Are Ready for Help. Please go to Knowlege-page for information retrieval !")
                        
                    except Exception as e:
                        print(f"A error occured while saving you files")
                        st.error(f"There was some error generating embeddings")
        else:
            st.info("Please upload PDF files to proceed.")
        
    with tab2:    
        st.header(":blue[Your Saved Documnets:]")
        saved_files = get_all_files()
        
        with st.container(border=True):
            refresh = st.button("",key="rerun_button",icon="🔄")
            if refresh:
                st.rerun()
            col1, col2 = st.columns([10, 2])  # Adjust column width ratios
            # Iterate over the saved files
            for file in saved_files:
                with col1:
                    file_name = f"{file["file_name"][:33].strip()}..." if len(file["file_name"])>33 else file["file_name"]
                    # Display the file name and file ID in a tabular format
                    st.markdown(f"<h4>🖇️<i>{file_name}</i></h4>",unsafe_allow_html=True)
                
                with col2:
                    # Create a delete button for each file using the file ID
                    if st.button("", key=file["file_id"],icon="🗑️"):
                        delete_file([file["file_id"]])
                        st.warning(f"Deleted File")
                        st.rerun()
                        # Add the deletion logic here
               

main()