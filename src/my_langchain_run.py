import sys
import os
import warnings

sys.path.append("./")
from scipy.spatial import distance
from src.my_langchain_agent import MyLangchainAgentHandler
from src.my_langchain_docs import MyLangchainDocsHandler

# suppress warnings for demo
warnings.filterwarnings("ignore")
os.environ["PYDEVD_INTERRUPT_THREAD_TIMEOUT"] = "60"


# select model and lora
model_name = "llama-7b"
lora_name = "alpaca-lora-7b"

testAgent = MyLangchainAgentHandler()
embedding = testAgent.load_hf_embedding()

text1 = "This is a very long sentence and the only difference is a period at the end"
text2 = "This is a very long sentence and the only difference is a period at the end."
query_result1 = embedding.embed_query(text1)
query_result2 = embedding.embed_query(text2)
print(f"text1 and text2 distance: {distance.euclidean(query_result1, query_result2)}")
doc_result = embedding.embed_documents([text1, text2])
print(f"text1 within doc:\n{str(doc_result[0][0:5]).replace(']','...')}")
print(f"text2 within doc:\n{str(doc_result[1][0:5]).replace(']','...')}")

# load llm
hf, model, tokenizer = testAgent.load_llama_llm(
    model_name=model_name, lora_name=lora_name, max_new_tokens=200
)

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
index = testDocs.load_docs_into_chroma(file_list, index_name)
index = testDocs.load_docs_into_redis(file_list, index_name)

# using the VectorStoreIndexWrapper to run a chain with question related to the doc
index.vectorstore.similarity_search(
    "What did the president say about Ketanji Brown Jackson"
)
query = "What did the president say about Ketanji Brown Jackson"
doc_response = index.query(query, llm=hf)
print(f"Query - {query}\nResponse - \n{doc_response}")

# using the VectorStoreIndexWrapper to run a chain with question related to the doc
query = "What did the president say about Ketanji Brown Jackson"
index.vectorstore.similarity_search(query)
retriever = index.vectorstore.as_retriever()
from langchain.chains import RetrievalQA

retrievalQA = RetrievalQA.from_llm(llm=model, retriever=retriever, verbose=True)

doc_response = index.query(question=query, llm=hf, kwargs={"verbose": True})
print(f"Query - {query}\nResponse - \n{doc_response}")

# Template for prompt
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationSummaryBufferMemory,
    ConversationBufferMemory,
)
from langchain.prompts.prompt import PromptTemplate
from src.my_langchain_agent import MyLangchainAgentHandler

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

# from langchain.llms import LlamaCppEmbeddings
# llm = LlamaCppEmbeddings(model_path="/path/to/llama/model")
