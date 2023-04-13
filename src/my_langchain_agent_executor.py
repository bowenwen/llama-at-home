# Goal: Agent with doc search

import logging
import sys
import os
import warnings

sys.path.append("./")

from langchain.utilities import (
    WikipediaAPIWrapper,
    SerpAPIWrapper,
    GoogleSearchAPIWrapper,
)
from langchain.chains import ConversationChain, RetrievalQA
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import BaseTool
from src.my_langchain_agent import MyLangchainAgentHandler
from src.my_langchain_docs import MyLangchainDocsHandler

# suppress warnings for demo
warnings.filterwarnings("ignore")
os.environ["PYDEVD_INTERRUPT_THREAD_TIMEOUT"] = "60"
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT "] = "60"


# set up log
# targets = logging.StreamHandler(sys.stdout), logging.FileHandler("test.log")
# logging.basicConfig(format="%(message)s", level=logging.INFO, handlers=targets)
# log = open("test.log", "a")
# sys.stdout = log
# print = log.info()

# select model and lora
model_name = "llama-13b"
# model_name = "llama-30b-4bit-128g"
lora_name = "alpaca-lora-13b"

testAgent = MyLangchainAgentHandler()
embedding = testAgent.load_hf_embedding()
hf, model, tokenizer = testAgent.load_llama_llm(
    model_name=model_name, lora_name=lora_name, max_new_tokens=50
)

# start simple gradio app
import gradio as gr


# index documents
index_name = "examples"

file_list_master = {
    "examples": ["index-docs/examples/state_of_the_union.txt"],
    "arxiv": ["index-docs/arxiv/2302.13971.pdf"],
    "psych_dsm": ["index-docs/psych/DSM-5-TR.pdf"],
    "psych_sop": ["index-docs/psych/Synopsis_of_Psychiatry.pdf"],
}
file_list = file_list_master[index_name]

testDocs = MyLangchainDocsHandler(embedding=embedding, redis_host="192.168.1.236")
# index = testDocs.load_docs_into_chroma(file_list, index_name)
index = testDocs.load_docs_into_redis(file_list, index_name)
vectorstore_retriever = index.vectorstore.as_retriever()
state_of_union = RetrievalQA.from_chain_type(
    llm=hf, chain_type="stuff", retriever=vectorstore_retriever
)

wikipedia = WikipediaAPIWrapper()


def truncate_wikipedia(input_string):
    full_string = wikipedia.run(input_string)
    truncated_string = full_string[0:1000]
    return truncated_string


def summarize_wikipedia(input_string):
    # use LLM to summarize the meaning of the Wikipedia text
    return None


def get_secrets(key_name):
    _key_file = open(f"secrets/{key_name}.key", "r", encoding="utf-8")
    _key_value = _key_file.read()
    return _key_value


# Search
# Serp API
# os.environ["SERPAPI_API_KEY"] = get_secrets("serpapi")
# search = SerpAPIWrapper()

# Google API
os.environ["GOOGLE_API_KEY"] = get_secrets("googleapi")
os.environ["GOOGLE_CSE_ID"] = get_secrets("googlecse")
search = GoogleSearchAPIWrapper()

# Try this instead:
# Google search
# https://python.langchain.com/en/latest/modules/agents/tools/examples/google_search.html

tools = [
    Tool(
        name="State of Union Document",
        func=state_of_union.run,
        description="contains the recent state of the union address",
    ),
    Tool(
        name="Wikipedia",
        func=truncate_wikipedia,  # replacing wikipedia.run
        description="useful for when you need to find facts and information from Wikipedia.",
    ),
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to find facts and information from Google.",
    ),
    Tool(
        name="Google",
        func=search.run,
        description="useful for when you need to find facts and information from Google.",
    ),
]

# tools = [
#     Tool(
#         name = "Intermediate Answer",
#         func=wikipedia.run,
#         description="useful for when you need to find information from Wikipedia."
#     )
# ]

# NOT WORKING:
## problem is that it is not properly calling the tools
## ideas to try:
## different AgentType?
## go back to examples?
## manually format things?

# main_prompt = """What year did Russia annexed Crimea? Did the president mention this historical event in the state of the union?"""

main_prompt = "Create a summary of current events and highlight ones related to the Economy."

agent = initialize_agent(
    tools, hf, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
agent.run(main_prompt)

print("done")

# def question_answer(question, not_used):
#     template = """Below is an instruction that describes a task.
#     {history}
#     ### Instruction:
#     {input}
#     ### Response:"""
#     prompt = PromptTemplate(input_variables=["history", "input"], template=template)
#     # Conversation chain
#     conversation = ConversationChain(
#         prompt=prompt,
#         llm=hf,
#         verbose=True,
#         memory=ConversationBufferMemory(),
#     )
#     with open("test.log", "r") as file:
#         langchain_log = file.read()
#     return [langchain_log, conversation.predict(input=question)]


# gr.Interface(
#     fn=question_answer,
#     inputs=["text", "text"],
#     outputs=["textbox", "textbox"],
# ).launch(server_name="0.0.0.0", server_port=7860)


# print("stop")


# # Template for prompt
# template = """Bob and a cat are having a friendly conversation.

# Current conversation:
# {history}
# Bob: {input}
# Cat:"""
# PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
# # Conversation chain
# conversation = ConversationChain(
#     prompt=PROMPT,
#     llm=hf,
#     verbose=True,
#     memory=ConversationBufferMemory(),
# )
# print(conversation.predict(input="Hi there!"))


# # Template for prompt with alpaha lora
# template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# {history}
# ### Instruction:
# {input}
# ### Response:"""
# PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
# # Conversation chain
# conversation = ConversationChain(
#     prompt=PROMPT,
#     llm=hf,
#     verbose=True,
#     memory=ConversationSummaryBufferMemory(llm=hf, max_token_limit=100),
# )
# response = conversation.predict(input="What is the biggest ocean in the world?")
# print(response)
# response = conversation.predict(input="What is the greatest scientific invention?")
# print(response)
# response = conversation.predict(input="What is the dealiest war?")
# print(response)
# response = conversation.predict(
#     input="What are some of the largest companies in the world?"
# )
# print(response)
# print("stop")
