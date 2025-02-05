from typing_extensions import List,TypedDict, Annotated,Union
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    user_query: Annotated[List, add_messages]
    llm_response:Annotated[List,add_messages]
    tool_to_use:Annotated[List, add_messages]
    query_for_tool:Annotated[List,add_messages]
    vector_results: Annotated[List, add_messages]
    web_results: Annotated[List, add_messages]
    search_arxiv:Annotated[List,add_messages]
    
    
    
