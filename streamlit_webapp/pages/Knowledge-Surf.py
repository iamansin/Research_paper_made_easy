import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from my_package.agents import ChatAgent
import os
import json
import time
import asyncio
from my_package.database_operations import get_all_sessions, create_session, delete_session,store_chat_message,get_chat_messages_by_session
import json
from dotenv import load_dotenv
load_dotenv()

# Use a pipeline as a high-level helper



groq_api = os.getenv("GROQ_API_KEY")
LLM  = ChatGroq(api_key=groq_api,model="llama-3.1-70b-versatile")
LLM_final = ChatGroq(api_key=groq_api, model="llama-3.1-70b-versatile",streaming=True)
agent = ChatAgent(_llm=LLM,_llm_final=LLM_final) 
Graph  =  agent.graph


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


async def workflow(query: str,session_id,start_time):
    user_query = HumanMessage(content=query)
    response = []
    try:
        # Use `astream` to get async generator
        res = Graph.astream({"user_query": user_query}, stream_mode="messages")
        async for event in res:
            node = event[1].get('langgraph_node')
            if node == "Final":
                response.append(event[0].content)
                yield event[0].content
                
        yield st.divider
        suggestions = []
        arxiv_results = agent.arxiv_results
        if arxiv_results:
            for value in arxiv_results:
                try:
                    content = (
                        f"**Title:** {value.get('Title', 'Unknown Title')}  \n"
                        f"**Summary:** {value.get('Summary', 'No summary available.')}  \n"
                        f"[{value.get('Link', '#')}]({value.get('Link', '#')})\n\n"
                    )
                    suggestions.append([content])
                    yield  content
                except Exception as e:
                    print(f"**Error fetching details for a paper:** {e}\n\n")
        response_time = round(time.time() - start_time, 2) 
        st.markdown(f"`⏱️{response_time} sec`")
        res = {'response': ''.join(response),'suggestions':suggestions,'response_time':response_time}
        str_res = json.dumps(res)
        store_chat_message(session_id, "assistant", str_res)
        
    except Exception as e:
        raise f"Error occurred while processing: {e}"
    

def sync_workflow(query: str, session_id,start_time):
    """
    Convert async generator to a synchronous generator for compatibility with st.write_stream.
    """
    async_gen = workflow(query,session_id,start_time)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Iterate over async generator using asyncio loop and yield synchronously
        while True:
            yield loop.run_until_complete(async_gen.__anext__())
    except StopAsyncIteration:
        pass
    finally:
        loop.close()
    
def main():
    
    #Side Bar Logic ----------------------------------------
    icon_add = [":heavy_plus_sign:"]
    icon_delete = [":material/delete:"]

    # Input for creating a new session
    if "add_new_chat" not in st.session_state:
        st.session_state["add_new_chat"] = False

    if "new_session" not in st.session_state:
        st.session_state["new_session"] = None
    
    if "selected_session" not in st.session_state:
        st.session_state["selected_session"] = None
    
    if "current_session" not in st.session_state:
        st.session_state["current_session"] = None
      
    add_new_chat = st.sidebar.pills("`New Chat`", options=icon_add, selection_mode="single", default=None,)

    if add_new_chat:
        session_name = st.sidebar.text_input("Chat name:").title()
        if st.sidebar.button("",icon="✔️"):
            if session_name.strip():
                st.session_state["new_session"] = create_session(session_name).id
                st.sidebar.success(f"Session '{session_name}' created!")
                st.session_state["add_new_chat"] = False
                st.rerun()
            else:
                st.sidebar.error("Session name cannot be empty!")

    # Display saved sessions with delete buttons
    sessions = get_all_sessions()
    
        
    if sessions:
        st.sidebar.markdown("*Saved Chats* :speech_balloon:")
        print(f"This is currently using session id for THE session--------->{st.session_state["selected_session"]}")
        for session in sessions[::-1]:
            col1, col2 = st.sidebar.columns([15, 1])
            with col1:
                if st.button(f"🌐 {session.name[:26].strip()}..", use_container_width=True, key=f"session_{session.id}",):
                    st.session_state["selected_session"] = session.id
                    print(f"This is current session id of SELECTED session--------->{st.session_state["selected_session"]}, session name :{session.name}")
                   
            with col2:
                if st.sidebar.pills(label="D", options="🗑️", key=f"delete_{session.id}",
                                    selection_mode="single", default=None, label_visibility='collapsed'):
                    delete_session(session.id)
                    st.sidebar.warning(f"Session '{session.name}' deleted!")
                    st.rerun()
    
    else:
        st.session_state["selected_session"] = create_session("First Chat").id
        print(f"This is current session id for the FIRST time interaction --------->{st.session_state["current_session"]}")
      
      
      
    # knowldege-Surf Logic -----------------------------------------
    
    st.title(":rainbow[Deep-dive] in your _papers_ here!")
    
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if st.session_state["new_session"]:
        st.session_state["current_session"] = st.session_state["new_session"]
        st.session_state["new_session"] = None
    elif st.session_state["selected_session"]:
        st.session_state["current_session"] = st.session_state["selected_session"]
        print(f"This is current{st.session_state["current_session"]} , the selected session is :{st.session_state["selected_session"]}")
    else:
        print(f"No selected session print the last one--->{sessions[-1].id}")
        st.session_state["current_session"] = sessions[-1].id
        
    session_id = st.session_state["current_session"]
    print(f"Loding chats for ------>{st.session_state.current_session}")
    st.session_state.messages = get_chat_messages_by_session(session_id)
    # Initialize chat history

    chat_container = st.container()
    # Render existing chat messages without moving the heading
    for message in st.session_state.messages:
        with chat_container:
            with st.chat_message(message["role"]):            
                if message["role"] == "assistant":
                    res_dict = json.loads(message["content"])   
                    response = res_dict.get('response') 
                    suggestions = res_dict.get('suggestions','')
                    response_time = res_dict.get('response_time')
                    print(suggestions)
                    if suggestions:
                        st.markdown(response)
                        st.markdown("----")
                        for s in suggestions:
                            st.markdown(s[0])
                    else:
                        st.markdown(response)
                    st.markdown(f"`⏱️{response_time} sec`")
                else:
                    st.markdown(message["content"])


    # Accept user input
    if prompt := st.chat_input("What is up?"):
        start_time = time.time()
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        print(f"Storing human chats to session------>{session_id}")
        print(f"Current session is-------->{st.session_state.current_session}")
        store_chat_message(session_id, "user", prompt)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner(text="Thinking..."):
                llm_response = st.write_stream(sync_workflow(prompt,session_id,start_time))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": llm_response})

main()
    