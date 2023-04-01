import os
import json
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

from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredPDFLoader

from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

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

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
hf = HuggingFacePipeline(pipeline=pipe)

# using the default embedding model from hf
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# test embeddings
# text = "This is a test document."
# query_result = embeddings.embed_query(text)
# doc_result = embeddings.embed_documents([text])
# example_embed1 = embeddings.embed_query("This is a very long sentence and the only difference is a period at the end")
# example_embed2 = embeddings.embed_query("This is a very long sentence and the only difference is a period at the end.")
# from scipy.spatial import distance
# print(f"{distance.euclidean(example_embed1, example_embed2)}")

# simple text gen
text = "Jim is a helpful business analyst that gives simple, practical answers to questions. \n Bob: What would be a good company name for a company that makes colorful socks? Give me a list of ideas. \n Jim: "
# text_response = hf(text)
# print(f"{text_response}\n")
# print(f"{'='*10}\n")
# text_response = hf(text)
# print(f"{text_response}\n")
# print(f"{'='*10}\n")

# index documents
loaded_doc_file = "docs/doc_loaded.json"
doc_list = [
    "docs/examples/state_of_the_union.txt",
    "docs/arxiv/2302.13971.pdf",
    # "docs/psych/DSM-5-TR.pdf",
    # "docs/psych/Synopsis_of_Psychiatry.pdf",
]


def smart_doc_load(doc_list, loaded_doc_file):
    loader_list = []
    for doc in doc_list:
        # get json record of docs loaded
        if os.path.exists(loaded_doc_file):
            with open(loaded_doc_file) as jfile:
                doc_loaded = json.loads(jfile.read())
        else:
            doc_loaded = []
        # check if doc has already been loaded, if so, skip
        if doc in doc_loaded:
            print(f"skipping {doc}")
        else:
            # load documents by type
            file_type = doc.split("/")[-1].split(".")[-1]
            if file_type == "txt":
                loader_list.append(TextLoader(doc))
            elif file_type == "pdf":
                loader_list.append(UnstructuredPDFLoader(doc))
            # save json record of docs loaded
            print(f"loaded {doc}")
            doc_loaded.append(doc)
            with open(loaded_doc_file, "w") as outfile:
                json.dump(doc_loaded, outfile)
    return loader_list


# https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/redis.html
loader_list = smart_doc_load(doc_list, loaded_doc_file)
vectorstore_kwargs = {"persist_directory": "./.chroma/persist"}
index = VectorstoreIndexCreator(
    embedding=embeddings, vectorstore_kwargs=vectorstore_kwargs
).from_loaders(loader_list)
print(f"processed {loader_list} into vector store")


# db = Chroma(embedding_function=embeddings, **vectorstore_kwargs)
# loaded_index = VectorStoreIndexWrapper(vectorstore=db)

# query = "What did the president say about Ketanji Brown Jackson"
# docs = db.similarity_search(query)

# ask some questions about the documents
query = "What did the president say about Ketanji Brown Jackson"
doc_response = index.query(query, llm=hf)
print(f"Query - {query}\nResponse - \n{doc_response}")

query = "What are some diagnostic features of substance use disorders?"
doc_response = index.query(query, llm=hf)
print(f"Query - {query}\nResponse - \n{doc_response}")

query = "What data sources are used by llama?"
doc_response = index.query(query, llm=hf)
print(f"Query - {query}\nResponse - \n{doc_response}")

query = "What method did llama use to tokenize its data?"
doc_response = index.query(query, llm=hf)
print(f"Query - {query}\nResponse - \n{doc_response}")

# index.query_with_sources(query)

# from langchain.chains.question_answering import load_qa_chain

# chain = load_qa_chain(hf, chain_type="stuff")
# chain.run(input_documents=docs, question=query)


# text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=100, chunk_overlap=0)
# texts = text_splitter.split_text(state_of_the_union)
