from pathlib import Path
from langchain.llms import HuggingFacePipeline
from transformers import (
    # AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter

# model_id = "gpt2"
model_name = "llama-7b"
model_path = f"models/{model_name}"

# TODO: review config.json with model folder:
# https://huggingface.co/docs/transformers/v4.27.2/en/internal/generation_utils#transformers.TemperatureLogitsWarper

tokenizer = AutoTokenizer.from_pretrained(Path(f"{model_path}/"))
tokenizer.truncation_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    Path(model_path),
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
)
# model = model.cuda()

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50)
hf = HuggingFacePipeline(pipeline=pipe)
embeddings = HuggingFaceEmbeddings()

# test embeddings
# text = "This is a test document."
# query_result = embeddings.embed_query(text)
# doc_result = embeddings.embed_documents([text])

# simple text gen
text = "Jim is a helpful business analyst that gives simple, practical answers to questions. \n Bob: What would be a good company name for a company that makes colorful socks? Give me a list of ideas. \n Jim: "
# text_response = hf(text)
# print(f"{text_response}\n")
# print(f"{'='*10}\n")
# text_response = hf(text)
# print(f"{text_response}\n")
# print(f"{'='*10}\n")

# do document retrieval
loader = TextLoader("./packages/langchain/docs/modules/state_of_the_union.txt")
# loader.load()[0]

# ask a question about the document
index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])
query = "What did the president say about Ketanji Brown Jackson"
doc_response = index.query(query, llm=hf)
# index.query_with_sources(query, llm=hf)
print(doc_response)

# from langchain.chains.question_answering import load_qa_chain

# chain = load_qa_chain(hf, chain_type="stuff")
# chain.run(input_documents=docs, question=query)


# text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=100, chunk_overlap=0)
# texts = text_splitter.split_text(state_of_the_union)
