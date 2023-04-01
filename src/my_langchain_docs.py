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


class MyLangchainRedisDocsHandler:
    """a wrapper to make loading my documents easier"""

    DEFAULT_DOMAIN = "localhost"

    def connect_to_redis():
        pass

    def load_docs_into_redis(
        self, embeddings, loaded_doc_file, doc_list, redis_host=DEFAULT_DOMAIN
    ):

        # https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/redis.html
        loader_list = self.smart_generate_doc_loaders(doc_list, loaded_doc_file)
        vectorstore_kwargs = {"persist_directory": "./.chroma/persist"}

        loader = TextLoader("../../../state_of_the_union.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        rds = Redis.from_documents(
            docs, embeddings, redis_url=f"redis://{redis_host}:6379", index_name="link"
        )
        # rds.index_name
        query = "What did the president say about Ketanji Brown Jackson"
        results = rds.similarity_search(query)
        print(results[0].page_content)

        # index = VectorstoreIndexCreator(
        #     embedding=embeddings, vectorstore_kwargs=vectorstore_kwargs
        # ).from_loaders(loader_list)
        print(f"processed {loader_list} into vector store")

    @staticmethod
    def smart_generate_doc_loaders(doc_list, loaded_doc_file):
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


if __name__ == "main":
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # index documents
    loaded_doc_file = "docs/doc_loaded.json"
    doc_list = [
        "docs/examples/state_of_the_union.txt",
        "docs/arxiv/2302.13971.pdf",
        # "docs/psych/DSM-5-TR.pdf",
        # "docs/psych/Synopsis_of_Psychiatry.pdf",
    ]
    tester = MyLangchainRedisDocsHandler()
    tester.load_docs_into_redis(
        loaded_doc_file=loaded_doc_file,
        doc_list=doc_list,
        redis_host="192.168.1.236",
        embeddings=embedding,
    )
