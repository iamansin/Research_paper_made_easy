from agents import ChatAgent 
from states import AgentState
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import asyncio

load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")

LLM  = ChatGroq(api_key=groq_api,model="llama-3.1-70b-versatile",temperature=0)

LLM_final = ChatGroq(api_key=groq_api, model="llama-3.1-70b-versatile",streaming=True)


async def workflow(query: str):
    # Initialize the ChatAgent with a provided LLM
    agent = ChatAgent(_llm=LLM, _llm_final=LLM_final)
    graph = agent.graph

    try:
        # Use `astream` to get async generator
        res = graph.astream({"user_query": HumanMessage(content=query)}, stream_mode="messages")

        # Use `async for` to iterate over the results
        async for event in res:
            node = event[1].get('langgraph_node')
            if node == "Final":
                print(event[0].content, end="")

        # Access the arXiv results after processing the stream
        print("\n")
        print("Results from arXiv -------------------->")
        arxiv_results = agent.arxiv_results
        if arxiv_results:
            for value in arxiv_results:
                content = f"Title:{value["Title"]}  Summary:{value["Summary"]}.........  {value["Link"]} "
                print(content)
        else:
            print("Got no results")
    except Exception as e:
        raise f"Error occurred while processing: {e}"

query = " How many encoder decoder pairs were used in the actual 'attention is all you need paper', do not show me some similar research papers "


asyncio.run(workflow(query))
# Generate response via streaming
# async def main():
#     query = " How many encoder decoder pairs were used in the actual 'attention is all you need paper', Also show me some similar research papers ?"
#     async for result in workflow(query):
#         if "llm_response" in result:
#             print(result,end="")
#         elif "research_papers" in result:
#             print(result)

# asyncio.run(main())

# def lol():
#     results = [{'Title': 'Hard-Coded Gaussian Attention for Neural Machine Translation', 'Link': 'http://arxiv.org/abs/2005.00742v1', 'Summary': "Recent work has questioned the importance of the Transformer's multi-headed\nattention for achieving ", 'Relevance Score': 0.6074}, {'Title': 'You May Not Need Attention', 'Link': 'http://arxiv.org/abs/1810.13409v1', 'Summary': 'In NMT, how far can we get without attention and without separate encoding\nand decoding? To answer t', 'Relevance Score': 0.5579}, {'Title': 'Attention Is Indeed All You Need: Semantically Attention-Guided Decoding for Data-to-Text NLG', 'Link': 'http://arxiv.org/abs/2109.07043v1', 'Summary': 'Ever since neural models were adopted in data-to-text language generation,\nthey have invariably been', 'Relevance Score': 0.5337}, {'Title': 'Attention Is All You Need for Chinese Word Segmentation', 'Link': 'http://arxiv.org/abs/1910.14537v3', 'Summary': 'Taking greedy decoding algorithm as it should be, this work focuses on\nfurther strengthening the mod', 'Relevance Score': 0.5016}, {'Title': 'ED2LM: Encoder-Decoder to Language Model for Faster Document Re-ranking Inference', 'Link': 'http://arxiv.org/abs/2204.11458v1', 'Summary': 'State-of-the-art neural models typically encode document-query pairs using\ncross-attention for re-ra', 'Relevance Score': 0.4636}]

#     for value in results:
#         content = f"Title:{value["Title"]}  Summary:{value["Summary"]}.........  {value["Link"]} "
#         yield content

# for val in lol():
#     print(val)

# def test_chat_agent(query: str):
#     agent = ChatAgent(_llm = LLM,_llm_final=LLM_final)                        
#     vector_results =ToolMessage(content=["(0.7510548824324277, 'itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding\\nlayers, produce outputs of dimension dmodel = 512 .\\nDecoder: The decoder is also composed of a stack of N= 6identical layers. In addition to the two\\nsub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head\\nattention over the output of the encoder stack. Similar to the encoder, we employ residual connections\\naround each of the sub-layers, followed by layer normalization. We also modify the self-attention\\nsub-layer in the decoder stack to prevent positions from attending to subsequent positions. This\\nmasking, combined with fact that the output embeddings are offset by one position, ensures that the\\npredictions for position ican depend only on the known outputs at positions less than i.\\n3.2 Attention\\nAn attention function can be described as mapping a query and a set of key-value pairs to an output,', {'page': 2, 'source': 'Attention is all you need.pdf'})", "(0.8456353266415385, '3.2 Attention\\nAn attention function can be described as mapping a query and a set of key-value pairs to an output,\\nwhere the query, keys, values, and output are all vectors. The output is computed as a weighted sum\\n3', {'page': 2, 'source': 'Attention is all you need.pdf'})", "(0.8534923839832388, 'encoder.\\n•Similarly, self-attention layers in the decoder allow each position in the decoder to attend to\\nall positions in the decoder up to and including that position. We need to prevent leftward\\ninformation flow in the decoder to preserve the auto-regressive property. We implement this\\ninside of scaled dot-product attention by masking out (setting to −∞) all values in the input\\nof the softmax which correspond to illegal connections. See Figure 2.\\n3.3 Position-wise Feed-Forward Networks\\nIn addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully\\nconnected feed-forward network, which is applied to each position separately and identically. This\\nconsists of two linear transformations with a ReLU activation in between.\\nFFN( x) = max(0 , xW 1+b1)W2+b2 (2)\\nWhile the linear transformations are the same across different positions, they use different parameters', {'page': 4, 'source': 'Attention is all you need.pdf'})", "(0.8729740799234861, 'orO(logk(n))in the case of dilated convolutions [ 18], increasing the length of the longest paths\\nbetween any two positions in the network. Convolutional layers are generally more expensive than\\nrecurrent layers, by a factor of k. Separable convolutions [ 6], however, decrease the complexity\\nconsiderably, to O(k·n·d+n·d2). Even with k=n, however, the complexity of a separable\\nconvolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer,\\nthe approach we take in our model.\\nAs side benefit, self-attention could yield more interpretable models. We inspect attention distributions\\nfrom our models and present and discuss examples in the appendix. Not only do individual attention\\nheads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic\\nand semantic structure of the sentences.\\n5 Training\\nThis section describes the training regime for our models.\\n5.1 Training Data and Batching', {'page': 6, 'source': 'Attention is all you need.pdf'})", "(0.8774405771998925, 'on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014\\nEnglish-to-French translation tasks, we achieve a new state of the art. In the former task our best\\nmodel outperforms even all previously reported ensembles.\\nWe are excited about the future of attention-based models and plan to apply them to other tasks. We\\nplan to extend the Transformer to problems involving input and output modalities other than text and\\nto investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs\\nsuch as images, audio and video. Making generation less sequential is another research goals of ours.\\nThe code we used to train and evaluate our models is available at https://github.com/\\ntensorflow/tensor2tensor .\\nAcknowledgements We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful\\ncomments, corrections and inspiration.\\nReferences\\n[1]Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint', {'page': 9, 'source': 'Attention is all you need.pdf'})"], tool_call_id='call_1db')
# #     # Prepare an initial agent state with an example query
#     # state: AgentState = {'user_query': [HumanMessage(content='how many encoder-decoder paires were used in the attention all you need paper?', additional_kwargs={}, response_metadata={}, id='990653af-0fec-46dd-b2a6-cb3e3b9db456')],
#     #                      'llm_response': [AIMessage(content='', additional_kwargs={}, response_metadata={}, id='af782de2-9c16-4e05-ae38-381bb206a14d')], 
#     #                      'tool_to_use': [AIMessage(content='search_vector_store', additional_kwargs={}, response_metadata={}, id='7265a38a-c12a-451d-af2c-4099011d774c')],
#     #                      'query_for_tool': [AIMessage(content='encoder-decoder pairs in attention all you need paper', additional_kwargs={}, response_metadata={}, id='7041fcf4-fea1-43fa-b0d5-4cd26412e95d')],
#     #                      'vector_results': [ToolMessage(content=["(0.7510548824324277, 'itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding\\nlayers, produce outputs of dimension dmodel = 512 .\\nDecoder: The decoder is also composed of a stack of N= 6identical layers. In addition to the two\\nsub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head\\nattention over the output of the encoder stack. Similar to the encoder, we employ residual connections\\naround each of the sub-layers, followed by layer normalization. We also modify the self-attention\\nsub-layer in the decoder stack to prevent positions from attending to subsequent positions. This\\nmasking, combined with fact that the output embeddings are offset by one position, ensures that the\\npredictions for position ican depend only on the known outputs at positions less than i.\\n3.2 Attention\\nAn attention function can be described as mapping a query and a set of key-value pairs to an output,', {'page': 2, 'source': 'Attention is all you need.pdf'})", "(0.8456353266415385, '3.2 Attention\\nAn attention function can be described as mapping a query and a set of key-value pairs to an output,\\nwhere the query, keys, values, and output are all vectors. The output is computed as a weighted sum\\n3', {'page': 2, 'source': 'Attention is all you need.pdf'})", "(0.8534923839832388, 'encoder.\\n•Similarly, self-attention layers in the decoder allow each position in the decoder to attend to\\nall positions in the decoder up to and including that position. We need to prevent leftward\\ninformation flow in the decoder to preserve the auto-regressive property. We implement this\\ninside of scaled dot-product attention by masking out (setting to −∞) all values in the input\\nof the softmax which correspond to illegal connections. See Figure 2.\\n3.3 Position-wise Feed-Forward Networks\\nIn addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully\\nconnected feed-forward network, which is applied to each position separately and identically. This\\nconsists of two linear transformations with a ReLU activation in between.\\nFFN( x) = max(0 , xW 1+b1)W2+b2 (2)\\nWhile the linear transformations are the same across different positions, they use different parameters', {'page': 4, 'source': 'Attention is all you need.pdf'})", "(0.8729740799234861, 'orO(logk(n))in the case of dilated convolutions [ 18], increasing the length of the longest paths\\nbetween any two positions in the network. Convolutional layers are generally more expensive than\\nrecurrent layers, by a factor of k. Separable convolutions [ 6], however, decrease the complexity\\nconsiderably, to O(k·n·d+n·d2). Even with k=n, however, the complexity of a separable\\nconvolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer,\\nthe approach we take in our model.\\nAs side benefit, self-attention could yield more interpretable models. We inspect attention distributions\\nfrom our models and present and discuss examples in the appendix. Not only do individual attention\\nheads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic\\nand semantic structure of the sentences.\\n5 Training\\nThis section describes the training regime for our models.\\n5.1 Training Data and Batching', {'page': 6, 'source': 'Attention is all you need.pdf'})", "(0.8774405771998925, 'on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014\\nEnglish-to-French translation tasks, we achieve a new state of the art. In the former task our best\\nmodel outperforms even all previously reported ensembles.\\nWe are excited about the future of attention-based models and plan to apply them to other tasks. We\\nplan to extend the Transformer to problems involving input and output modalities other than text and\\nto investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs\\nsuch as images, audio and video. Making generation less sequential is another research goals of ours.\\nThe code we used to train and evaluate our models is available at https://github.com/\\ntensorflow/tensor2tensor .\\nAcknowledgements We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful\\ncomments, corrections and inspiration.\\nReferences\\n[1]Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint', {'page': 9, 'source': 'Attention is all you need.pdf'})"], tool_call_id='call_1db')],
#     #                      'web_results': [],
#     #                      'arxiv_results': [ToolMessage(content=[{'Title': 'Fastformer: Additive Attention Can Be All You Need', 'Link': 'http://arxiv.org/abs/2108.09084v6', 'Summary': 'Transformer is a powerful model for text understanding. However, it is\ninefficient due to its quadratic complexity to input sequence length. Although\nthere are many methods on Transformer acceleration', 'Relevance Score': 0.485}, {'Title': 'Three Sentences Are All You Need: Local Path Enhanced Document Relation Extraction', 'Link': 'http://arxiv.org/abs/2106.01793v1', 'Summary': 'Document-level Relation Extraction (RE) is a more challenging task than\nsentence RE as it often requires reasoning over multiple sentences. Yet, human\nannotators usually use a small number of sentence', 'Relevance Score': 0.3739}, {'Title': 'Object Counting: You Only Need to Look at One', 'Link': 'http://arxiv.org/abs/2112.05993v1', 'Summary': 'This paper aims to tackle the challenging task of one-shot object counting.\nGiven an image containing novel, previously unseen category objects, the goal\nof the task is to count all instances in the d', 'Relevance Score': 0.3247}, {'Title': 'Grounding is All You Need? Dual Temporal Grounding for Video Dialog', 'Link': 'http://arxiv.org/abs/2410.05767v2', 'Summary': 'In the realm of video dialog response generation, the understanding of video\ncontent and the temporal nuances of conversation history are paramount. While a\nsegment of current research leans heavily o', 'Relevance Score': 0.3004}, {'Title': 'The Matrix Calculus You Need For Deep Learning', 'Link': 'http://arxiv.org/abs/1802.01528v3', 'Summary': 'This paper is an attempt to explain all the matrix calculus you need in order\nto understand the training of deep neural networks. We assume no math knowledge\nbeyond what you learned in calculus 1, and', 'Relevance Score': 0.1945}], tool_call_id='call_1arxiv')]}
    
#     uq = HumanMessage(content=query)
#     state:AgentState = {
#         "user_query":[],
#         "arxiv_results":[ToolMessage(content=[{'Title': 'Fastformer: Additive Attention Can Be All You Need', 'Link': 'http://arxiv.org/abs/2108.09084v6', 'Summary': 'Transformer is a powerful model for text understanding. However, it is\ninefficient due to its quadratic complexity to input sequence length. Although\nthere are many methods on Transformer acceleration', 'Relevance Score': 0.485}, {'Title': 'Three Sentences Are All You Need: Local Path Enhanced Document Relation Extraction', 'Link': 'http://arxiv.org/abs/2106.01793v1', 'Summary': 'Document-level Relation Extraction (RE) is a more challenging task than\nsentence RE as it often requires reasoning over multiple sentences. Yet, human\nannotators usually use a small number of sentence', 'Relevance Score': 0.3739}, {'Title': 'Object Counting: You Only Need to Look at One', 'Link': 'http://arxiv.org/abs/2112.05993v1', 'Summary': 'This paper aims to tackle the challenging task of one-shot object counting.\nGiven an image containing novel, previously unseen category objects, the goal\nof the task is to count all instances in the d', 'Relevance Score': 0.3247}, {'Title': 'Grounding is All You Need? Dual Temporal Grounding for Video Dialog', 'Link': 'http://arxiv.org/abs/2410.05767v2', 'Summary': 'In the realm of video dialog response generation, the understanding of video\ncontent and the temporal nuances of conversation history are paramount. While a\nsegment of current research leans heavily o', 'Relevance Score': 0.3004}, {'Title': 'The Matrix Calculus You Need For Deep Learning', 'Link': 'http://arxiv.org/abs/1802.01528v3', 'Summary': 'This paper is an attempt to explain all the matrix calculus you need in order\nto understand the training of deep neural networks. We assume no math knowledge\nbeyond what you learned in calculus 1, and', 'Relevance Score': 0.1945}], tool_call_id='call_1arxiv')],
#         "llm_response":[],
#         "query_for_tool":[],
#         "tool_to_use":[],
#         "vector_results":[],
#         "web_results":[]
#     }
#     state["user_query"].append(uq)

#     logger.info(f"Running test with query----->{query}")
#     print("-----------------------------------------------------------------------------------------------")
#     # # Test the flow for processing the query through the agent's graph
#     state1 = agent.Chat_bot(state=state)
#     logger.info("State after processing------------->\n")
#     logger.info(state1)
#     print("-----------------------------------------------------------------------------------------------")
    
#     path = agent.decide_path(state1)
#     logger.info(path)
#     print("-----------------------------------------------------------------------------------------------")
    
#     if path == "end":
#         state4 = agent.final_response(state=state1)
#         res = state4["llm_response"][-1].content
#         logger.info("The final response that will be streamed ---->")
#         logger.info(res)
#         return res
    
#     state2= state1['vector_results'].append(vector_results)
#     logger.info("State after processing TOOL---->Added to state")
#     print("-----------------------------------------------------------------------------------------------")
#     # Output the final response after all the tool processing and routing
    
    
    
#     state3 = agent.Chat_bot(state=state2)
#     logger.info("State after TOOL and LLM Response ,2nd pass to the Chat Bot-->")
#     logger.info(state1)
#     print("-----------------------------------------------------------------------------------------------")
    
#     path = agent.decide_path(state3)
#     logger.info(path)
    
#     print("-----------------------------------------------------------------------------------------------")
    
#     state4 = agent.final_response(state=state3)
#     logger.info("The final response that will be streamed ---->")
#     logger.info(state4)
    
# #     # Assuming `state["llm_response"]` contains the final response after all tool processing
# #     final_response = final_state.get("llm_response", "No response generated")
# #     print("Final Response Generated by the Agent:")
# #     print(final_response)

# if __name__ =="__main__":
#     user_query ="How many encoder decoder pairs were used in the actual 'attention is all you need paper'?"
#     test_chat_agent(user_query)

# import json
# import re

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

# # Example usage
# content='```json\n{\n    "tool": "end",\n    "tool_query": "How many encoder-decoder pairs were used in the \'Attention is all you need\' paper?",\n    "response": "This document does not specify the number of encoder-decoder pairs used in the \'Attention is all you need\' paper."\n}\n``` \n'
# print(parse_json_response(content))