from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage,SystemMessage,ToolMessage,HumanMessage
from langchain.prompts import ChatPromptTemplate
from  typing_extensions import TypedDict,Literal,Optional,Union
import json
import re
import asyncio
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from my_package.tools import TOOLS
from my_package.states import AgentState
from my_package.logger import setup_colored_logging
from my_package.prompts import SYSTEM_PROMPT, VECTORS_STORE_TOOL_PROMPT, WEB_SEARCH_TOOL_PROMPT, FINAL_PROMPT
from my_package.arxiv_search import search_arxiv
# from my_package.memory_store import get_context_from_memory,Neo4jHandler,update_memory
# from my_package.database_operations import store_chat_message
logger = setup_colored_logging(name=__name__)

class StructuredResponse(BaseModel):
    tool:Literal["search_vector_store", "search_web", "end"] = Field(description="This field indicates the tool used to generate the output.The options are : ['search_vector_store','search_web','end']")
    tool_query:Optional[str] = Field(description="This field contains a very short and concise version of the user query used for querying the tools")
    response:Optional[str] = Field(description="This field contains the response from the LLM if and only if there is any need to generate any response")
    use_arxiv_search:Optional[Union[bool,Literal["Fasle", "True"]]] = Field(description="This field indicates whether there is any need to use arxiv search tool, if yes -'True' otherwise- 'False'")
    
    

# #For structured output...
# class JSONStructOutput(TypedDict):
#     tool: Literal["search_vector_store", "search_web", "end"]
#     tool_query: Optional[str]
#     response: Optional[str]
#     use_arxiv_search: Optional[Union[bool,Literal["Fasle", "True"]]]


# def parse_json_response(content: str) -> dict:
#     """
#     Parses a string containing JSON-like content and converts it to a Python dictionary.
    
#     Parameters:
#         content (str): The input string containing JSON-like content.
    
#     Returns:
#         dict: A dictionary representation of the JSON content.
#     """
#     try:
#         # Extract the JSON part between the code block markers
#         match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
#         if not match:
#             raise ValueError("No JSON code block found in the content.")
        
#         json_content = match.group(1)  # Extract the JSON string
#         parsed_data = json.loads(json_content)  # Parse it into a dictionary
#         return parsed_data
#     except json.JSONDecodeError as e:
#         raise ValueError(f"Error parsing JSON: {e}")
#     except Exception as e:
#         raise ValueError(f"Error: {e}")



#definig agents ...
class ChatAgent: 
    def __init__(self,tools=TOOLS, _llm=None,_llm_final = None):
        # Initialize tools
        self._tools = {t.name: t for t in tools}
        self._llm = _llm
        self._llm_final = _llm_final
        self.graph = self.graph_builder()
        self.arxiv_task = None
        self.arxiv_results = None
        # self.memory_graph = Neo4jHandler()
        
        
    def graph_builder(self):
        # Set up LangGraph StateGraph
        builder = StateGraph(AgentState)
        builder.add_node("Chat_bot", self.Chat_bot)
        builder.add_node("search_vector_store", self.search_vector_store)
        builder.add_node("search_web", self.search_web)
        # builder.add_node("arxiv",self.arxiv_search)
        builder.add_node("Final",self.final_response)
        
        #entry point
        builder.set_entry_point("Chat_bot")
        
        # Conditional routing from route_query
        builder.add_conditional_edges("Chat_bot", self.decide_path, {
            "vector_search": "search_vector_store",
            "web_search": "search_web",
            "end":"Final"
        })
    
        # builder.add_edge("Chat_bot","arxiv")
        builder.add_edge("search_vector_store","Chat_bot")
        builder.add_edge("search_web","Chat_bot")
        builder.add_edge("Final", END)
        # Compile the graph 
        graph = builder.compile()
        return graph

    async def Chat_bot(self, state: AgentState):
        """Initial node to route user query based on the type of request."""
        
        if state["tool_to_use"]:
            logger.info("Input query to appropriate tool...")
            response_from_tool = state["tool_to_use"][-1].content
            logger.debug(f"Got response from the Tool----> {response_from_tool}")
            Template = None
            Previous_context = state["llm_response"][-1] if state["llm_response"] else "IGNORE"
            Context_from_tool = None
            
            if response_from_tool == "search_vector_store":
                logger.debug(f"Entering to ----> {response_from_tool}")
                Context_from_tool = state["vector_results"][-1]
                Template = VECTORS_STORE_TOOL_PROMPT
                
            else:
                logger.debug(f"Entering to ----> {response_from_tool}")
                Context_from_tool = state["web_results"][-1]
                Template = WEB_SEARCH_TOOL_PROMPT
                
            parser = PydanticOutputParser(pydantic_object=StructuredResponse)
            prompt = PromptTemplate(
                                    template=Template,
                                    partial_variables={"format_instructions": parser.get_format_instructions()},
                                )
            message = prompt.format(previous_context=Previous_context,context_from_tool=Context_from_tool)
            logger.info("Now invoking LLM for response...")
            try:
                llm_response = self.get_structured_response(message=message)
            except Exception as e:
                logger.error(f"Got Error {e} ,after serveral retires--> could not get response from LLM")
                raise e
            try:      
                if llm_response:
                    logger.warning(f"Response type ---->{type(llm_response)}")
                    logger.warning(f"The response got ---------->{llm_response}")
                    
                    tool = llm_response.get("tool", "end")
                    response = llm_response.get("response", "Not able to get response")
                    logger.debug(f"The tool value is ---->{tool}")
                    state["tool_to_use"].append(AIMessage(tool))  # Append selected tool to state
                    
                    # Ensure response is a string before appending
                    logger.debug(f"The response value is ---->{response}")
                    if isinstance(response, str):
                        state["llm_response"].append(AIMessage(content=response))
                    else:
                        logger.error(f"Unexpected response type: {type(response)}. Value: {response}")
                else:
                    raise ValueError("Not able to get Structured response from the LLM!!")
            except Exception as e:
                logger.error(f"An Error occured, falling back to END , the error is :{e}")
                logger.critical("Falling back to default--> __END__ ")
                state["tool_to_use"] = [AIMessage("end")]   
                
        else:
            logger.debug("---------------------------------------------------------------------------")
            logger.info("Starting Chat Agent.")
            user_query = state["user_query"][-1]
            parser = PydanticOutputParser(pydantic_object=StructuredResponse)
            prompt = PromptTemplate(
                                    template=SYSTEM_PROMPT,
                                    partial_variables={"format_instructions": parser.get_format_instructions()},
                                )
            message = prompt.format(query=user_query)
            # prompt_template = ChatPromptTemplate.from_messages([
            #                         SystemMessage(content=SYSTEM_PROMPT),
            #                         HumanMessage(content=f"{query}")
            #                         ])
            # message = prompt_template.format_messages()
            logger.info("Invoking LLM for response...")
            try:
                llm_response = self.get_structured_response(message=message)
            except Exception as e:
                logger.error(f"Got Error ,after serveral retires-->{e}")
                raise e         
            try:
                logger.warning(f"Response type ---->{type(llm_response)}")
                # logger.warning(f"The response got ---------->{llm_response}")
                
                if llm_response:
                    # Extract tool and default to "end" if not found
                    tool = llm_response.tool
                    # response = llm_response.get("response", "Not able to get response")
                    query_for_tool = llm_response.tool_query
                    use_arxiv_search = llm_response.use_arxiv_search
                    
                    logger.debug(f"The tool value is ---->{tool}")
                    state["tool_to_use"].append(AIMessage(tool))  # Append selected tool to state
                    
                    # Ensure response is a string before appending
                    # logger.debug(f"The response value is ---->{response}")
                    # if isinstance(response, str):
                    #     state["llm_response"].append(AIMessage(content=response))
                    # else:
                    #     logger.error(f"Unexpected response type: {type(response)}. Value: {response}")
                    
                    # Handle query_for_tool safely
                    if query_for_tool:
                        logger.debug(f"The query_for_tool value is ---->{query_for_tool}")
                        state["query_for_tool"].append(AIMessage(content=query_for_tool))
                    else:
                        logger.warning("query_for_tool is None or empty.")
                    
                    # Safely handle use_arxiv_search as boolean or string
                    logger.debug(f"Use arXiv ----> {use_arxiv_search}")
                    if not isinstance(use_arxiv_search,str):
                        state["search_arxiv"].append(AIMessage(str(use_arxiv_search)))
                    else:
                        state["search_arxiv"].append(AIMessage(use_arxiv_search))
                    
                    if use_arxiv_search in [True,"true","True"]:
                            query = (
                                state["query_for_tool"][0].content
                                if state["query_for_tool"]
                                else state["user_query"][0].content
                                    )
                            query = query[:200] if len(query)>200 else query
                            
                            try:
                                if not self.arxiv_task:  # Ensure only one task runs
                                    logger.info("Entering async loop......")
                                    self.arxiv_task = asyncio.create_task(self.get_arxiv_results(query))
                                else:
                                    logger.info("arXiv search already running.")
                                    
                                logger.info("aync function is now over.")
                            except Exception as e:
                                logger.critical(f"Error while running arXiv search: {e}") 
                    
                    else:
                        logger.info(f"Skipping arXiv search as search flag {use_arxiv_search}.")
                
            except Exception as e:
                logger.error(f"An Error occurred, falling back to END. Error: {e}")
                logger.warning("Falling back to default --> __END__")
                state["tool_to_use"].append(AIMessage("end"))   
                          
        return state

    def get_structured_response(self, message: str):
        """
        Get a structured response from the LLM with retries.
        
        Args:
            message (str): The input message to the LLM.

        Returns:
            JSONStructOutput: The structured output from the LLM if successful.
        
        Raises:
            RuntimeError: If all retry attempts fail.
        """
        llm_structured = self._llm.with_structured_output(StructuredResponse)
        
        for attempt in range(1,4):
            try:
                logger.info(f"Attempt {attempt}: Invoking LLM")
                structured_response = llm_structured.invoke(message)
                logger.warning(f"The response recieved from LLM in structured format ---> {structured_response}")
                if structured_response:
                    return structured_response
                # else:
                #     unstructured_reponse = self._llm.invoke(message)
                #     logger.warning("Invoking Non structural response")
                #     logger.info(f"Response without structed format {unstructured_reponse}")
                #     res = parse_json_response(unstructured_reponse.content)
                #     if res:
                #         return res
                #     else:
                #         continue
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {str(e)}")
                logger.info("Retrying...")
        checker = self._llm.invoke(message)
        logger.info(f"Response without structed format {checker}")
        # If all retries fail, raise an exception
        logger.critical("Failed to get a structured response from the LLM after 3 attempts.")
        raise RuntimeError("Failed to get a structured response from the LLM after 3 attempts.")
    
    async def get_arxiv_results(self,query:str):
        try:
            logger.info("Inside async search_arxiv function")
            results = await search_arxiv(query)
            return results
        except Exception as e:
            logger.critical(f"An Error occured while running asyncronous search_arxiv ---->{e}")
            return None
        
        
    def decide_path(self, state: AgentState):
        """
        Logic to decide the path based on user query content.
        
        Args:
            state (AgentState): The current state containing decision parameters.
        
        Returns:
            str: The path to follow based on the tool to use.
        """
        tool = state['tool_to_use'][-1].content
        logger.info(f"Deciding path based on the tool: {tool}")
        if tool == "search_web":
            logger.debug("Path decided:---->web_search")
            return "web_search"
        elif tool == "search_vector_store":
            logger.debug("Path decided:----> vector_search")
            return "vector_search"
        else:
            logger.info(f"{state}")
            logger.debug("Path decided:----> end")
            return "end"

    def search_vector_store(self, state: AgentState):
        """Node to handle vector store search."""
        query = state['query_for_tool'][-1].content
        logger.info("Using SEARCH VECTOR Tool...")
        try:
            result = self._tools["search_vector_store"].func(query)
            state['vector_results'].append(ToolMessage(content=result,tool_call_id='call_1db'))
            logger.info("Got results from ChromaDB")
        except Exception as e:
            logger.error(f"Error while getting vector results: {e}")
            state['vector_results'].append("Some problem getting documents from the database")
        return state

    def search_web(self, state: AgentState):
        """Node to handle web search for explanations."""
        query = state['query_for_tool'][-1].content
        logger.info("Using SEARCH WEB Tool...")
        try:
            result = self._tools["search_web"].func(query)
            state['web_results'].append(ToolMessage(content=result,tool_call_id='call_1web'))
            logger.info("Got results from web")
        except Exception as e:
            logger.error(f"Error searching web: {e}")
            state['web_results'].append("No results found")
        return state
    
    async def final_response(self,state:AgentState):
        logger.info("Into Final Node..")
        
     #If the self.arxiv_task is exists and it's done the results will be saved to self.arxiv_resutls    
        if self.arxiv_task:
            try:
                # Await the task to ensure it completes
                logger.warning("The Task is still running")
                if self.arxiv_task.done():
                    logger.info("The task is done and value stored in self.arxiv_results")
                    self.arxiv_results = self.arxiv_task
                else:
                    logger.info("Awaitng for the Task to get completed ")
                    self.arxiv_results = await self.arxiv_task
                logger.info(f"arXiv results retrieved")
                self.arxiv_task = None  # Reset the task
            except Exception as e:
                logger.error(f"Error retrieving arXiv results: {e}")
                self.arxiv_task = None    
                
                
        messages = state["llm_response"]
        query = state["user_query"][0].content
        last_ai_message = messages
        if len(messages)>0:
            last_ai_message = messages[-1].content
        else:
            last_ai_message = "NOT ABLE TO GET PREVIOUS RESPONSE"
        logger.info("Invoking final response")
        attempt = 1
        while attempt<4 :
                logger.info(f"Attempt {attempt}: Invoking LLM ")
                prompt= FINAL_PROMPT.format(context=last_ai_message,user_query=query)
                message  = [SystemMessage(content=prompt)]
                streamed_responses = []
                try:
                    response_stream = self._llm_final.stream(message)
                    for chunk in response_stream:
                        chunk_content = chunk.content.strip()
                        if chunk_content:  # Ignore empty chunks
                            streamed_responses.append(chunk_content)
                            yield {"llm_response": chunk_content} # Stream each chunk
                    logger.info("Successfully received response from LLM.")
                    break
                
                except Exception as e:
                    logger.error(f"Attempt {attempt} failed: {str(e)}")
                    logger.info("Retrying...")
                    attempt +=1
                    continue
                
        if attempt >= 3:
            logger.critical("Failed to get a structured response from the LLM after 3 attempts.")
            raise RuntimeError("Failed to get a structured response from the LLM after 3 attempts.")
        
    
    # def memory_retriever(self,state:AgentState,config):
    #     session_id = config.get("configurable").get("session_id")
    #     user_query = state["user_query"]
    #     get_context_from_memory(db =self.memory_graph ,state=state,session_id=session_id,top_k=5)
    #     return None 

    # def memory_processor(self,state:AgentState,config):
    #     session_id = config.get("configurable").get("session_id")
    #     message_id = store_chat_message(session_id=session_id,).id
        
        


        
