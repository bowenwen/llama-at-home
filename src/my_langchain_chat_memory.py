import sys
import os
import warnings

sys.path.append("./")

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate
from src.my_langchain_agent import MyLangchainAgentHandler
from src.my_langchain_docs import MyLangchainDocsHandler

# suppress warnings for demo
warnings.filterwarnings("ignore")
os.environ["PYDEVD_INTERRUPT_THREAD_TIMEOUT"] = "60"


# select model and lora
model_name = "llama-7b"
lora_name = "alpaca-lora-7b"

testAgent = MyLangchainAgentHandler(lora_name=lora_name)
embedding = testAgent.load_hf_embedding()
hf, model, tokenizer = testAgent.load_llama_llm(
    model_name=model_name, max_new_tokens=50
)

# Template for prompt
template = """Bob and a cat are having a friendly conversation.

Current conversation:
{history}
Bob: {input}
Cat:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
# Conversation chain
conversation = ConversationChain(
    prompt=PROMPT,
    llm=hf,
    verbose=True,
    memory=ConversationSummaryBufferMemory(llm=hf, max_token_limit=20),
)
conversation.predict(input="Hi there!")


# Template for prompt with alpaha lora
template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please continue the conversation

### Input:
{history}
{input}

### Response:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
# Conversation chain
conversation = ConversationChain(
    prompt=PROMPT,
    llm=hf,
    verbose=True,
    memory=ConversationSummaryBufferMemory(llm=hf, max_token_limit=20),
)
conversation.predict(input="Hi there!")
