import os
import json
from typing import Any, List, Optional, Type

from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredPDFLoader

from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.redis import Redis
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)


def _get_default_text_splitter() -> TextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


class MyLangchainDocsHandler:
    """a wrapper to make loading my documents easier"""

    DEFAULT_DOMAIN = "localhost"
    CHROMA_DIR = "./.chroma"
    DOC_DIR = "./index-docs"

    def __init__(self, embedding, redis_host=None):
        self.embedding = embedding
        if redis_host == None:
            redis_host = self.DEFAULT_DOMAIN
        self.redis_host = redis_host
        # db is the current doc database
        self.db = None
        self.text_splitter = _get_default_text_splitter()

    def index_from_redis(self, index_name):
        self.db = Redis.from_existing_index(
            self.embedding,
            redis_url=f"redis://{self.redis_host}:6379",
            index_name=index_name,
        )
        return VectorStoreIndexWrapper(vectorstore=self.db)

    def load_docs_into_redis(self, file_list, index_name):
        # only load files that do not already exist, otherwise, load from existing index
        filtered_file_list = self.filter_file_list(file_list, index_name, "Redis")
        if len(filtered_file_list) == 0:
            return self.index_from_redis(index_name)
        # load new docs into index
        vectorstore_kwargs = {"redis_url": f"redis://{self.redis_host}:6379"}
        self.db = self.build_vectorstore_from_loaders(
            loader_list=self.build_loader_list(file_list=filtered_file_list),
            vectorstore_cls=Redis,
            vectorstore_kwargs=vectorstore_kwargs,
        )
        return VectorStoreIndexWrapper(vectorstore=self.db)

    def index_from_chroma(self, index_name):
        vectorstore_kwargs = {"persist_directory": f"{self.CHROMA_DIR}/{index_name}"}
        self.db = Chroma(
            embedding_function=self.embedding,
            **vectorstore_kwargs,
        )
        return VectorStoreIndexWrapper(vectorstore=self.db)

    def load_docs_into_chroma(self, file_list, index_name):
        # only load files that do not already exist, otherwise, load from existing index
        filtered_file_list = self.filter_file_list(file_list, index_name, "Chroma")
        if len(filtered_file_list) == 0:
            return self.index_from_chroma(index_name)

        # load new docs into index
        vectorstore_kwargs = {"persist_directory": f"{self.CHROMA_DIR}/{index_name}"}
        self.db = self.build_vectorstore_from_loaders(
            loader_list=self.build_loader_list(file_list=filtered_file_list),
            vectorstore_cls=Chroma,
            vectorstore_kwargs=vectorstore_kwargs,
        )
        self.db.persist()
        return VectorStoreIndexWrapper(vectorstore=self.db)

    def build_vectorstore_from_loaders(
        self,
        loader_list: list,
        vectorstore_cls: Type[VectorStore],
        vectorstore_kwargs: dict = None,
    ) -> Type[VectorStore]:
        """building vectorstore db object from a list of document loaders

        Args:
            loader_list (list): document loaders
            vectorstore_cls (Type[VectorStore]): vector store objects
            vectorstore_kwargs (dict, optional): extra arguments for vectorstores. Defaults to {}.

        Returns:
            Type[VectorStore]: vectorstore object
        """
        docs = []
        for loader in loader_list:
            docs.extend(loader.load())
        sub_docs = self.text_splitter.split_documents(docs)
        if vectorstore_kwargs is None:
            vectorstore = vectorstore_cls.from_documents(sub_docs, self.embedding)
        else:
            vectorstore = vectorstore_cls.from_documents(
                sub_docs, self.embedding, **vectorstore_kwargs
            )
        return vectorstore

    def build_loader_list(self, file_list: list[str]) -> list[object]:
        """build a list of loaders based on document types

        Args:
            file_list (list[str]): list of paths to documents

        Returns:
            list[object]: list of loaders
        """

        # load documents by type
        loader_list = []
        for file_path in file_list:
            file_type = file_path.split("/")[-1].split(".")[-1]
            if file_type == "txt":
                loader_list.append(TextLoader(file_path))
            elif file_type == "pdf":
                loader_list.append(UnstructuredPDFLoader(file_path))
        return loader_list

    def filter_file_list(
        self, file_list: list[str], index_name: str, db_type: str
    ) -> list[str]:
        """filter the list of document by removing any documents previously
        had been indexed and recorded in {index_name}.json files

        TODO: this function currently assumes we are loading all docs into the index once
        need to add support and test ability to load new docs into previous indices

        Args:
            file_list (list[str]): list of paths to documents
            index_name (str): name of index
            db_type (str): name of db type like Chroma or Redis

        Returns:
            list[str]: list of filtered file list
        """
        filtered_file_list = []
        for file_path in file_list:
            # get json record of docs loaded
            index_record_file = f"{self.DOC_DIR}/db_{db_type}_{index_name}.json"
            index_record_file = os.path.relpath(index_record_file)
            if os.path.exists(index_record_file):
                with open(index_record_file, "r", encoding="utf-8") as infile:
                    filtered_file_list = json.loads(infile.read())
            else:
                filtered_file_list = []
            # check if doc has already been loaded, if so, skip
            if file_path in filtered_file_list:
                print(f"skipping {file_path}")
            else:
                # save json record of docs loaded
                print(f"loaded {file_path}")
                filtered_file_list.append(file_path)
                with open(index_record_file, "w", encoding="utf-8") as outfile:
                    json.dump(filtered_file_list, outfile)
        return filtered_file_list
