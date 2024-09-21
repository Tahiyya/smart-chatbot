from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
import pandas as pd
from tqdm import tqdm

MODEL_NAME = "phi3:mini"

reader = SimpleDirectoryReader(input_dir="Data", recursive=True)
docs = reader.load_data()

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = Ollama(
    model=MODEL_NAME,
    temperature=0.2,
    request_timeout=360.0,
    system_prompt = "You are an AI language model trained to answer questions on a product called - Milli-Q® IQ 7000 Ultrapure Water System. It is a product of M, a Life Science company. You are given the user question and the external knowledge based on product specifications to answer that question. Keep your answers concise and based on facts – do not hallucinate features. If you do not know, respond by saying that you do not know."
)

index = VectorStoreIndex.from_documents(docs)
rag_application = index.as_chat_engine(chat_mode="context", verbose=True, streaming=True)

def query_chat_get_response(user_input):
    # LlamaIndex returns a response object that contains both the output string and retrieved nodes
    response_object = rag_application.chat(user_input)

    
    # Process the response object to get the output string and retrieved nodes
    if response_object is not None:
        actual_output = response_object.response
        retrieval_context = [node.get_content() for node in response_object.source_nodes]
        return actual_output, retrieval_context
    return None, []

# user_list = ["Does the Milli-Q® IQ 7000 remove endotoxins from water?", "What is the resistivity of the water produced by the Milli-Q® IQ 7000?"]
df = pd.read_excel(r"C:\Users\tahiy\VS Code Scripts\Chatbot-Retrieval\Evaluation\Eval_Dataset.xlsx")

actual_output_list = []
retreival_final_list1 = []
retreival_final_list2 = []
# Iterate over each row in the 'Questions' column
for index, row in tqdm(df.iterrows(), desc="Generating answers to User Queries", total=len(df)):
    question = row["Question"]
    actual_output_str, retreival_list = query_chat_get_response(question)
    actual_output_list.append(actual_output_str)
    retreival_final_list1.append(retreival_list[0])
    retreival_final_list2.append(retreival_list[1])

# df = pd.DataFrame()
# df['Question'] = user_list
df['Actual Output'] = actual_output_list
df['Retreived Context 1'] = retreival_final_list1
df['Retreived Context 2'] = retreival_final_list2

df.to_excel(r"C:\Users\tahiy\VS Code Scripts\Chatbot-Retrieval\Evaluation\Eval_Dataset_Output.xlsx")

# LlamaIndex returns a response object that contains both the output string and retrieved nodes

# Process the response object to get the output string and retrieved nodes
# if response_object is not None:
#     actual_output = response_object.response
#     retrieval_context = [node.get_content() for node in response_object.source_nodes]

#  sources List[ToolOutput], unformatted_response str, List[NodeWithScore] source nodes

