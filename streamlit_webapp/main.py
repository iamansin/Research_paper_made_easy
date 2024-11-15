import streamlit as st


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.title("main page")

# def knowledge_surf_page():
#     st.title("Knowledge Surf")
#     papers = collection.find()

#     for paper in papers:
#         st.write(paper["title"])
#         st.write(paper["abstract"])

# def vector_db_page():
#     st.title("VectorDB")
#     vectors = milvus.search_vectors("embeddings", "query", "Euclidean")
#     for vector in vectors:
#         st.write(vector)

# def context_page():
#     st.title("Context")
#     st.write("This is the context page")