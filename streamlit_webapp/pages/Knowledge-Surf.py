import streamlit as st
from my_package.chroma_manager import ChromaManager
from langchain.chains import history_aware_retriever

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def generated_response():
    chroma_manager = ChromaManager(persist_directory="chroma_database", collection_name="research_papers")
    yield "hello im assistant"




def main():
    st.title(":rainbow[Deep-dive] in your _papers_ here!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()

    # Render existing chat messages without moving the heading
    for message in st.session_state.messages:
        with chat_container:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(generated_response())
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    

main()
    