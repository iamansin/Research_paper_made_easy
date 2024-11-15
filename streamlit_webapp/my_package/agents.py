from langgraph.graph import StateGraph, END
from langchain.schema import  HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from  typing_extensions import TypedDict,Literal
import json
from tools import TOOLS
from states import AgentState
from prompts import SYSTEM_PROMPT,VECTORS_STORE_TOOL_PROMPT,ARXIV_TOOL_PROMPT,WEB_SEARCH_TOOL_PROMPT,FINAL_PROMPT


#For structured output...
class Action(TypedDict):
    tool: Literal["search_vector_store", "search_web", "search_arxiv", "end"]
    confidence_level: Literal["very-low", "low", "moderate", "high", "very-high"]
    explainability: Literal["simple", "complex"]
    response: str 

class JSONStructOutput(TypedDict):
    action: Action
    

#definig agents ...
class ChagAgent: 
    def __init__(self,tools=TOOLS, _llm=None):
        # Initialize tools
        self._tools = {t.name: t for t in tools}
        self._llm = _llm
        self.graph = self.graph_builder
        
        
        
    def graph_builder(self):
        # Set up LangGraph StateGraph
        builder = StateGraph(AgentState)
        builder.add_node("Chat_bot", self.Chat_bot)
        builder.add_node("search_vector_store", self.search_vector_store)
        builder.add_node("search_web", self.search_web)
        builder.add_node("search_arxiv", self.search_arxiv)
        builder.add_node("END", self.save_final_result)
        
        #entry point
        builder.set_entry_point("Chat_bot")
        
        # Conditional routing from route_query
        builder.add_conditional_edges("Chat_bot", self.decide_path, {
            "vector_search": "search_vector_store",
            "web_search": "search_web",
            "arxiv_search": "search_arxiv",
            "end":"END"
        })
        builder.add_edge("search_vector_store","Chat_bot")
        builder.add_edge("search_web","Chat_bot")
        builder.add_edge("search_arxiv","Chat_bot")
        builder.add_edge("END",END)
        # Compile the graph
        graph = builder.compile()
        return graph

    def Chat_bot(self, state: AgentState, query: str):
        """Initial node to route user query based on the type of request."""
        print("Input query to appropriate tool...")
        llm_structured = self._llm.with_structured_output(JSONStructOutput)
        
        if len(state["user_query"]) > 0 :
            response_from_tool = state["tool_to_use"][-1]
            print(f"Got response from the Tool {response_from_tool}")
            
            if response_from_tool == "search_vector_store":
                context_from_tool = state["vector_results"][-1]
                previous_response = state["llm_response"][-1]
                prompt_template = ChatPromptTemplate.from_template(VECTORS_STORE_TOOL_PROMPT)

            elif response_from_tool == "search_web":
                context_from_tool = state["web_results"][-1]
                previous_response = state["llm_response"][-1]
                prompt_template = ChatPromptTemplate.from_template(WEB_SEARCH_TOOL_PROMPT)

            elif response_from_tool == "search_arixv" :
                context_from_tool = state["arxiv_results"][-1]
                previous_response = state["llm_response"][-1]
                prompt_template = ChatPromptTemplate.from_template(ARXIV_TOOL_PROMPT)

            else:
                context_from_tool = state["arxiv_results"]
                previous_response = state["llm_response"][-1]
                prompt_template = ChatPromptTemplate.from_template(FINAL_PROMPT)
                
            message = prompt_template.format_messages(
                                            context_from_tool=context_from_tool,
                                            previous_response=previous_response)
            llm_response = llm_structured.invoke(message)
            parsed_response = llm_response 
            tool = parsed_response.get("action",{}).get("tool","end")
            response = parsed_response.get("action",{}).get("response","Not able to get response")
            state["tool_to_use"] = tool
            state["llm_response"] = response

        else:
            state["user_query"] = query
            prompt_template = ChatPromptTemplate.from_messages([
                                    SystemMessage(content=SYSTEM_PROMPT),
                                    HumanMessage(content=f"{query}")
                                    ])
            messages = prompt_template.format_messages()
            try:
                llm_response = llm_structured.invoke(messages)
                print(llm_response)
            
            except Exception as e:
                # Log the exact error message
                print(f"Error while getting the response from LLM: {e}")
                # Raising a ValueError with a more descriptive message for debugging
                raise ValueError("The LLM did not generate a valid response. Please check the prompt format.")
    
            try:
                parsed_response = llm_response 
                tool = parsed_response.get("action", {}).get("tool", "end")  
                response = parsed_response.get("action",{}).get("response","Not able to get response")
                print(f"The Tool Suggested {tool}")
                state["tool_to_use"] = tool  # Store selected tool in state
                state["llm_response"] = response
            except json.JSONDecodeError:
                print("Error: Could not parse LLM response JSON.")  # Safe fallback in case of JSON parsing error
                state["tool_to_use"] = "end" # Safe fallback in case of JSON parsing error 
                
        return state

    def decide_path(self, state: AgentState):
        """Logic to decide the path based on user query content."""
        tool= state['tool_to_use'][-1]
        if tool == "search_web":
            return "web_search"
        elif tool == "search_arixv":
            return "arxiv_search"
        elif tool == "search_vector_store":
            return "vector_search"
        else:
            return "end"

    def search_vector_store(self, state: AgentState):
        """Node to handle vector store search."""
        query = state['user_query']
        print("Using SEARCH VECTOR Tool......")
        try:
            state['vector_results'] = self._tools["search_vector_store"].func(query)
            print(f"got results from chroma db :{state["vector_results"][-1]}")
        except Exception as e:
            print(f"There was some error while getting vector results::{e}")
            state['vector_results']  = "Some problem getting documents from the database"
        return state

    def search_web(self, state: AgentState):
        """Node to handle web search for explanations."""
        query = state['user_query']
        print("Using SEARCH WEB Tool.....")
        try:
            state['web_results'] = self._tools["search_web"].func(query)
            print(f"got results from web :{state['web_results'][-1]}")
        except Exception as e:
            print(f"there was some error searching web:{e}")
            state['web_results'] = "No results found"
        return state

    def search_arxiv(self, state: AgentState):
        """Node to handle arXiv search for similar papers."""
        query = state['user_query']
        print("Using SEARCH ARXIV Tool......")
        try:
            state['arxiv_results'] = self._tools["arxiv_search"].func(query)
            print(f"Got results from arxiv :{state['arxiv_results'][-1]}")
        except Exception as e:
            print(f"there was some error getting similar papaers from arxiv {e}")
            state['arxiv_results']= "Not able to get results from the arxiv"
        return state
    
    def save_final_result(self, state):
    
        result = state["llm_response"][-1]
        print(f" the final response generated by the LLM: {result}")
        return state  # State remains intact for final access
