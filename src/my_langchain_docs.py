import os
import json

from langchain.embeddings import HuggingFaceEmbeddings

from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredPDFLoader

from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.redis import Redis


class MyLangchainDocsHandler:
    """a wrapper to make loading my documents easier"""

    DEFAULT_DOMAIN = "localhost"
    CHROMA_DIR = "./.chroma"
    DOC_DIR = "./docs"

    def __init__(self, embeddings, redis_host=None):
        self.embeddings = embeddings
        if redis_host == None:
            redis_host = self.DEFAULT_DOMAIN
        self.redis_host = redis_host

    def index_from_redis(self, index_name):
        rds = Redis.from_existing_index(
            self.embeddings,
            redis_url=f"redis://{self.redis_host}:6379",
            index_name=index_name,
        )
        return rds

    def load_docs_into_redis(self, doc_list, index_name):

        # https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/redis.html

        loader = TextLoader("../../../state_of_the_union.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        rds = Redis.from_documents(
            docs,
            self.embeddings,
            redis_url=f"redis://{self.redis_host}:6379",
            index_name=index_name,
        )
        # rds.index_name
        query = "What did the president say about Ketanji Brown Jackson"
        results = rds.similarity_search(query)
        print(results[0].page_content)

    def load_docs_into_chroma(self, doc_list, index_name):

        filtered_doc_list = self.filter_doc_list(doc_list, index_name)

        # load documents by type
        loader_list = []
        for doc in filtered_doc_list:
            file_type = doc.split("/")[-1].split(".")[-1]
            if file_type == "txt":
                loader_list.append(TextLoader(doc))
            elif file_type == "pdf":
                loader_list.append(UnstructuredPDFLoader(doc))

        vectorstore_kwargs = {"persist_directory": f"{self.CHROMA_DIR}/{index_name}"}

        index = VectorstoreIndexCreator(
            embedding=self.embeddings, vectorstore_kwargs=vectorstore_kwargs
        ).from_loaders(loader_list)

        return index

    def filter_doc_list(self, doc_list, index_name):
        filtered_doc_list = []
        for doc in doc_list:
            # get json record of docs loaded
            index_record_file = f"{self.DOC_DIR}/{index_name}.json"
            if os.path.exists(index_record_file):
                with open(index_record_file, "r") as jfile:
                    filtered_doc_list = json.loads(jfile.read())
            else:
                filtered_doc_list = []
            # check if doc has already been loaded, if so, skip
            if doc in filtered_doc_list:
                print(f"skipping {doc}")
            else:
                # save json record of docs loaded
                print(f"loaded {doc}")
                filtered_doc_list.append(doc)
                with open(index_record_file, "w") as outfile:
                    json.dump(filtered_doc_list, outfile)
        return filtered_doc_list


if __name__ == "__main__":

    # # index documents
    # index_name = "docs/doc_loaded.json"
    # doc_list = [
    #     "docs/examples/state_of_the_union.txt",
    #     "docs/arxiv/2302.13971.pdf",
    #     # "docs/psych/DSM-5-TR.pdf",
    #     # "docs/psych/Synopsis_of_Psychiatry.pdf",
    # ]
    # tester = MyLangchainDocsHandler(embeddings=embedding, redis_host="192.168.1.236")
    # # index = tester.load_docs_into_redis(
    # #     doc_list=doc_list,
    # #     index_name=index_name,
    # #     embeddings=embedding,
    # #
    # # )
    # index = tester.load_docs_into_chroma(
    #     doc_list=doc_list, index_name=index_name
    # )
    # query = "What did the president say about Ketanji Brown Jackson"
    # doc_response = index.query(query, llm=hf)
    # print(f"Query - {query}\nResponse - \n{doc_response}")
    pass
