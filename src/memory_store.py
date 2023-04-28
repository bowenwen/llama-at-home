import logging
import sys
import os
import warnings

import gradio as gr

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.pgvector import PGVector

sys.path.append("./")
from src.models import LlamaModelHandler
from src.docs import DocumentHandler
from langchain.text_splitter import TextSplitter

from src.util import get_secrets
from src.util import get_default_text_splitter


# suppress warnings for demo
warnings.filterwarnings("ignore")
os.environ["PYDEVD_INTERRUPT_THREAD_TIMEOUT"] = "60"
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT "] = "60"
os.environ["MYLANGCHAIN_SAVE_CHAT_HISTORY"] = "0"


class MemoryStore:
    """set up a long term memory handling for an agent executor to use as document tools"""

    CONNECTION_STRING = None
    MEMORY_COLLECTION = None
    MEMORY_COLLECTION_META = None
    DOC_EMBEDDING = None
    PG_VECTOR_STORE = None

    def __init__(self, embedding, memory_collection = "long_term"):
        _postgres_host = get_secrets("postgres_host")
        if _postgres_host is not None:
            os.environ["PGVECTOR_HOST"] = _postgres_host.split(":")[0]
            os.environ["PGVECTOR_PORT"] = _postgres_host.split(":")[1]
        _postgres_db = get_secrets("postgres_db")
        if _postgres_db is not None:
            os.environ["PGVECTOR_DATABASE"] = _postgres_db
        _postgres_user = get_secrets("postgres_user")
        if _postgres_user is not None:
            os.environ["PGVECTOR_USER"] = _postgres_user
        _postgres_pass = get_secrets("postgres_pass")
        if _postgres_pass is not None:
            os.environ["PGVECTOR_PASSWORD"] = _postgres_pass
        self.CONNECTION_STRING = PGVector.connection_string_from_db_params(
            driver=os.environ.get("PGVECTOR_DRIVER", default="psycopg2"),
            host=os.environ.get("PGVECTOR_HOST", default="localhost"),
            port=int(os.environ.get("PGVECTOR_PORT", default="5432")),
            database=os.environ.get("PGVECTOR_DATABASE", default="postgres"),
            user=os.environ.get("PGVECTOR_USER", default="postgres"),
            password=os.environ.get("PGVECTOR_PASSWORD", default="postgres"),
        )
        self.MEMORY_COLLECTION = memory_collection
        self.MEMORY_COLLECTION_META = {"description": "all of the long term memory stored as part of langchain agent runs."}
        self.DOC_EMBEDDING = embedding
        self.PG_VECTOR_STORE = PGVector(
            connection_string=self.CONNECTION_STRING,
            embedding_function=self.DOC_EMBEDDING ,
            collection_name=self.MEMORY_COLLECTION,
            collection_metadata=self.MEMORY_COLLECTION_META)

    def add_memory(self, text):
        # build texts object for memory 
        memory_texts = [text]
        # build
        # method 1: build custom text
        self.PG_VECTOR_STORE.add_texts(texts=memory_texts, metadatas=json_metadata, ids=["123"])

        # method 2: use splitter and from documents
        # splitter = get_default_text_splitter("character")
        # new_memory_doc = splitter.create_documents(texts = memory_texts, metadatas=json_metadata)
        # new_memory_doc_split = splitter.split_documents([new_memory_doc])
        # db = PGVector.from_documents(
        #     embedding=self.DOC_EMBEDDING,
        #     documents=new_memory_doc_split,
        #     collection_name=self.MEMORY_COLLECTION,
        #     connection_string=self.CONNECTION_STRING,
        # )

        pass

    def retrieve_memory(self, query):
        # add filter = {'type': 'reflection'}
        # example to get metadata for first item from similarity_search_with_score
        # get memory - result[0][0].page_content
        # get meta data - result[0][0].metadata
        # scale similarity - similarity_scale = result[0][1]
        pass
        