import logging
import sys
import os
import warnings

sys.path.append("./")

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate
from src.my_langchain_models import MyLangchainLlamaModelHandler
from src.my_langchain_docs import MyLangchainDocsHandler

# suppress warnings for demo
warnings.filterwarnings("ignore")
os.environ["PYDEVD_INTERRUPT_THREAD_TIMEOUT"] = "60"
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT "] = "60"
os.environ["MYLANGCHAIN_SAVE_CHAT_HISTORY"] = "1"

# select model and lora
# model_name = "llama-7b"
model_name = "llama-7b-4bit-128g"
lora_name = "alpaca-lora-7b"

testAgent = MyLangchainLlamaModelHandler()
embedding = testAgent.load_hf_embedding()
hf, model, tokenizer = testAgent.load_llama_llm(
    model_name=model_name, lora_name=lora_name, max_new_tokens=200
)

# start simple gradio app
import gradio as gr


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
#     answer = conversation.predict(input=question)
#     with open("output_now.log", "r") as file:
#         # optioanlly, read external log output from langchain
#         # require modifying packages/langchain/langchain/input.py
#         langchain_log = file.read()
#     with open("output_recent.log", "a") as f:
#         print(f"{answer}\n======", file=f)
#     return [langchain_log, answer]

template = """Below is an instruction that describes a task. 
{history}
### Instruction: {input}
### Response:"""
prompt = PromptTemplate(input_variables=["history", "input"], template=template)
# Conversation chain
conversation = ConversationChain(
    prompt=prompt,
    llm=hf,
    verbose=True,
    memory=ConversationBufferMemory(
        human_prefix="### Instruction", ai_prefix="### Response"
    ),
)


def question_answer_with_memory(question):
    answer = conversation.predict(input=question)
    with open("logs/output_now.log", "r") as file:
        # optioanlly, read external log output from langchain
        # require modifying packages/langchain/langchain/input.py
        langchain_log = file.read()
    if os.getenv("MYLANGCHAIN_SAVE_CHAT_HISTORY") == "1":
        with open("logs/output_recent.log", "a") as f:
            print(f"{answer}\n", file=f)
    return [langchain_log, answer]


gr.Interface(
    fn=question_answer_with_memory,
    inputs=["text"],
    outputs=["textbox", "textbox"],
).launch(server_name="0.0.0.0", server_port=7860)


print("stop")


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
