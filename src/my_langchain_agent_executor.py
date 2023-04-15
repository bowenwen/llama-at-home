# %%

# Goal: Agent with doc search

import logging
import sys
import os
import warnings

os.chdir("/home/coder/projects/llama-at-home")
sys.path.append("./")

from langchain.utilities import (
    WikipediaAPIWrapper,
    SerpAPIWrapper,
    GoogleSearchAPIWrapper,
    SearxSearchWrapper,
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
# lora_name = "alpaca-lora-13b"
# lora_name = "alpaca-gpt4-lora-13b"
lora_name = "alpaca-gpt4-lora-13b-3ep"

testAgent = MyLangchainAgentHandler()
embedding = testAgent.load_hf_embedding()
hf, model, tokenizer = testAgent.load_llama_llm(
    model_name=model_name, lora_name=lora_name, max_new_tokens=200
)

# %%
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

# %%
wikipedia = WikipediaAPIWrapper(top_k_results=1)


def truncate_wikipedia(input_string):
    full_string = wikipedia.run(input_string)
    truncated_string = full_string[0:2000]
    return truncated_string


def summarize_wikipedia(input_string):
    # use LLM to summarize the meaning of the Wikipedia text
    full_string = wikipedia.run(input_string)
    summarize_prompt = f"### Instruction: Please provide a detailed summary of the following information \n### Input:\n{full_string} \n### Response: "
    summarized_text = hf(summarize_prompt)
    return summarized_text


def get_secrets(key_name):
    _key_file = open(f"secrets/{key_name}.key", "r", encoding="utf-8")
    _key_value = _key_file.read()
    return _key_value


# # Search
# # Serp API
# os.environ["SERPAPI_API_KEY"] = get_secrets("serpapi")
# search = SerpAPIWrapper()

# # Google API
# os.environ["GOOGLE_API_KEY"] = get_secrets("google2api")
# os.environ["GOOGLE_CSE_ID"] = get_secrets("google2cse")
# search = GoogleSearchAPIWrapper(k=3)  # top k results only

# Searx API
# https://python.langchain.com/en/latest/modules/agents/tools/examples/searx_search.html
ssearch = SearxSearchWrapper(
    searx_host=get_secrets("searx_host"), k=3, engines=["google"]
)


def searx_google_search(input_string):
    # reference: https://searx.github.io/searx/admin/engines.html
    # https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/web_base.html
    truncated_string = ""
    while truncated_string == "":
        try:
            truncated_string = ssearch.run(input_string).replace("\n", " ")
        except:
            truncated_string = ""
    return truncated_string


# %%
# Try this instead:
# Google search
# https://python.langchain.com/en/latest/modules/agents/tools/examples/google_search.html

tools = [
    # Tool(
    #     name="State of Union QA System",
    #     func=state_of_union.run,
    #     description="specific facts from the 2023 state of the union on Joe Biden's plan to rebuild the economy and unite the nation.",
    # ),
    Tool(
        name="Wikipedia",
        func=summarize_wikipedia,
        # func=truncate_wikipedia,
        description="general information and well-established facts on a topic.",
    ),
    # Tool(
    #     name="Google",
    #     func=search.run,
    #     description="recent events and specific facts about a topic.",
    # ),
    Tool(
        name="Google",
        func=searx_google_search,
        description="recent events and specific facts about a topic.",
    ),
]

# enter prompt
# main_prompt = """What is the current inflation rate in the United States?"""
# main_prompt = """Who leaked the document on Ukraine?"""
# main_prompt = """Which city will be hosting the summer olympics in 2036?"""
# main_prompt = """Which city will be hosting the summer olympics in 2032?"""
# main_prompt = """What is the current progress on nuclear fusion?"""
# main_prompt = """What wars are happening around the world right now?"""
main_prompt = """Summarize the current events today on the US Economy."""
# main_prompt = """What is the current financial situation of TransLink?"""
print(
    "\x1b[1;32m"
    f"""\n\nQuestion: {main_prompt}
Thought:"""
    + "\x1b[0m"
)

# run agent executor chain
agent = initialize_agent(
    tools, hf, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
result = agent.run(main_prompt)
print(result)

# exit
print("done")

# %%
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
