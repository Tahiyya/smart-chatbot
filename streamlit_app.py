import streamlit as st
import openai
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)  #INFO
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# 💭📱🖥️🤖

st.set_page_config(page_title="Product Pro", page_icon="📱", layout="centered", initial_sidebar_state="auto", menu_items=None)
# openai.api_key = st.secrets.openai_key
st.title("Ask the Product Pro📱")
st.info("Chat with Product Pro, your AI powered Companion!", icon="💬")
# Welcome message with a lighter tone
st.markdown("""
    Welcome to the **Smart Product Information Chatbot**!  
            
    Ask me anything about product features and specifications.  
    Type your question below, and I’ll get you the most relevant information.
""")

MODEL_NAME = "phi3:mini"
# llm = Ollama(model="phi3:mini")

# print("Client setup")
# client = openai.OpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="nokeyneeded",
# )



if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Millipore Water System Products!",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    print("\n\nLoads data")
    reader = SimpleDirectoryReader(input_dir="Data", recursive=True)
    docs = reader.load_data()
    # Settings.llm = llm
    
    # bge-base embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")


    Settings.llm = Ollama(
        model=MODEL_NAME,
        # base_url="http://localhost:11434/v1",
        temperature=0.2,
        request_timeout=360.0,
        # system_prompt
        # prompt_key="""You are an expert on 
        # the Apple iPhone 13 features and your 
        # job is to answer questions. 
        # Assume that all questions are related 
        # to the iPhone 13. Keep 
        # your answers concise and based on 
        # facts – do not hallucinate features.""",
        # system_prompt="""You are an expert on 
        # the Apple iPhone 14 features and your 
        # job is to answer questions only related to the iPhone. 
        # Assume that all questions are related 
        # to the iPhone 14. Keep 
        # your answers concise and based on 
        # facts – do not hallucinate features.
        # If you do not know, respond saying that you do not know. """,
        system_prompt = "You are an AI language model trained to answer questions on a product called - Milli-Q® IQ 7000 Ultrapure Water System. It is a product of M, a Life Science company. You are given the user question and the external knowledge based on product specifications to answer that question. Keep your answers concise and based on facts – do not hallucinate features. If you do not know, respond by saying that you do not know."
    )
    index = VectorStoreIndex.from_documents(docs)
    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        # chat_mode="condense_question", verbose=True, streaming=True
        # chat_mode="condense_plus_context", verbose=True, streaming=True
        # condense_plus_context 
         
        chat_mode="context", verbose=True, streaming=True
    )
    # prompts_dict = st.session_state.chat_engine.get_prompts()
    # print(prompts_dict)
    
# prompts_dict = query_engine.get_prompts()
# display_prompt_dict(prompts_dict)

print("\n\nPost initializing = ", st.session_state.messages)

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

print("\n\nAfter query = ", st.session_state.messages)

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        # with st.spinner("Thinking..."):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)
        print("\n\nAfter response = ", st.session_state.messages)

# Testing
# query_engine = index.as_query_engine(similarity_top_k=2, llm=gpt35_llm)
# # use this for testing
# vector_retriever = index.as_retriever(similarity_top_k=2)
