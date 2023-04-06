import sys

sys.path.append("./")

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate
from src.my_langchain_agent import MyLangchainAgentHandler

# Load model
testAgent = MyLangchainAgentHandler()
embedding = testAgent.load_hf_embedding()
hf, model, tokenizer = testAgent.load_llama_llm(
    model_name="llama-7b", max_new_tokens=50
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
