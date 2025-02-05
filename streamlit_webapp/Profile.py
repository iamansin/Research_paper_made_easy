import streamlit as st
from my_package.database_operations import get_all_sessions

# Streamlit setup
st.set_page_config(
    page_title="Papers Made Easy",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': 'A personal project by Aman'},
)

# Sidebar: Create or select a chat session
#Side Bar Logic ----------------------------------------
def main():
    icon_add = [":material/add:"]
    # icon_delete = [":material/delete:"]
    if "add_new_chat" not in st.session_state:
        st.session_state["add_new_chat"] = False

    add_new_chat = st.sidebar.pills("`New Chat`", options=icon_add, selection_mode="single", default=None)

    if add_new_chat:
        st.sidebar.info("Open in Knowledge-Sruf")

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
                   
            # with col2:
            #     if st.sidebar.pills(label="D", options=icon_delete, key=f"delete_{session.id}",
            #                         selection_mode="single", default=None, label_visibility='collapsed'):
            #         delete_session(session.id)
            #         st.sidebar.warning(f"Session '{session.name}' deleted!")
            #         st.rerun()
    
    # else:
    #     st.session_state["selected_session"] = create_session("First Chat").id
    #     print(f"This is current session id for the FIRST time interaction --------->{st.session_state["current_session"]}")



main()