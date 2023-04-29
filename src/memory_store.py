# import logging
import sys
import os
import warnings
import re
import math
import hashlib

# import gradio as gr
import sqlalchemy
from sqlalchemy.orm import Session
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.pgvector import EmbeddingStore, PGVector

sys.path.append("/")
from src.models import LlamaModelHandler
from src.docs import DocumentHandler
from src.prompt import IMPORTANCE_RATING_PROMPT
from langchain.text_splitter import TextSplitter

from src.util import get_secrets, get_default_text_splitter, get_epoch_time

# suppress warnings for demo
warnings.filterwarnings("ignore")
os.environ["PYDEVD_INTERRUPT_THREAD_TIMEOUT"] = "60"
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT "] = "60"
os.environ["MYLANGCHAIN_SAVE_CHAT_HISTORY"] = "0"


class MemoryStore:
    """set up a long term memory handling for an agent executor to use as document tools"""

    connection_string = None
    memory_collection = None
    memory_collection_meta = None
    doc_embedding = None
    pg_vector_store = None

    def __init__(self, embedding, memory_collection="long_term"):
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
        self.connection_string = PGVector.connection_string_from_db_params(
            driver=os.environ.get("PGVECTOR_DRIVER", default="psycopg2"),
            host=os.environ.get("PGVECTOR_HOST", default="localhost"),
            port=int(os.environ.get("PGVECTOR_PORT", default="5432")),
            database=os.environ.get("PGVECTOR_DATABASE", default="postgres"),
            user=os.environ.get("PGVECTOR_USER", default="postgres"),
            password=os.environ.get("PGVECTOR_PASSWORD", default="postgres"),
        )
        self.memory_collection = memory_collection
        self.memory_collection_meta = {
            "description": "all of the long term memory stored as part of langchain agent runs."
        }
        self.doc_embedding = embedding
        self.pg_vector_store = PGVector(
            connection_string=self.connection_string,
            embedding_function=self.doc_embedding,
            collection_name=self.memory_collection,
            collection_metadata=self.memory_collection_meta,
        )

    def update_memory_access_time(self, custom_id):
        new_access_time = str(get_epoch_time())

        engine = sqlalchemy.create_engine(self.connection_string)
        conn = engine.connect()
        session = Session(conn)

        filter_by = EmbeddingStore.cmetadata["custom_id"].astext == custom_id

        memory_record = session.query(EmbeddingStore).filter(filter_by)
        memory_record_cmetadata = memory_record[0].cmetadata
        memory_record_cmetadata["access_time"] = new_access_time
        memory_record[0].cmetadata = memory_record_cmetadata
        session.commit()

    def add_memory(self, text, llm, with_importance=False):
        # build texts object for memory
        memory_texts = [text]
        store_time = str(get_epoch_time())
        access_time = store_time

        if with_importance:
            importance_rating_text = llm(
                IMPORTANCE_RATING_PROMPT.replace("{memory}", text)
            )
            re_result = re.search(r"\d", importance_rating_text)  # search for a digit
            if re_result:
                # if match is found
                importance_rating = int(re_result.group()) / 10
            else:
                # if no match is found
                importance_rating = 0.2
        else:
            importance_rating = 0.2

        # generate hash for id
        id_text = f"{store_time}_{text}"
        id_hash = hashlib.md5(id_text.strip().encode()).hexdigest()

        # build metadata
        json_metadata = [
            {
                "custom_id": id_hash,
                "store_time": store_time,
                "access_time": access_time,
                "instruction": "",
                "input": "",
                "output": "",
                "type": "",
                "importance": importance_rating,
            }
        ]

        # method 1: build custom text
        self.pg_vector_store.add_texts(
            texts=memory_texts, metadatas=json_metadata, ids=[id_hash]
        )

        # method 2: use splitter and from documents
        # splitter = get_default_text_splitter("character")
        # new_memory_doc = splitter.create_documents(texts = memory_texts, metadatas=json_metadata)
        # new_memory_doc_split = splitter.split_documents([new_memory_doc])
        # db = PGVector.from_documents(
        #     embedding=self.doc_embedding,
        #     documents=new_memory_doc_split,
        #     collection_name=self.memory_collection,
        #     connection_string=self.connection_string,
        # )

    def retrieve_top_memory(self, query):
        # access_time=str(get_epoch_time())
        # add filter = {"type": "reflection"}
        result = self.pg_vector_store.similarity_search_with_score(query, k=1)
        memory_content = result[0][0].page_content
        memory_metadata = result[0][0].metadata
        distance = result[0][1]
        # convert distance to similarity score
        similarity_score = 1 - ((1 / (1 + math.exp((-distance + 1) * 3))))

        # example to get metadata for first item from similarity_search_with_score
        # get memory - result[0][0].page_content
        # get meta data - result[0][0].metadata
        # scale similarity - similarity_scale = result[0][1]
        return [memory_content, similarity_score, memory_metadata]


if __name__ == "__main__":
    model_name = "llama-13b"
    lora_name = "alpaca-gpt4-lora-13b-3ep"
    testAgent = LlamaModelHandler()
    eb = testAgent.get_hf_embedding()

    pipeline, model, tokenizer = testAgent.load_llama_llm(
        model_name=model_name, lora_name=lora_name, max_new_tokens=200
    )

    memory_store = MemoryStore(embedding=eb)
    memory_store.add_memory("Test memory", llm=pipeline)
    memory_store.retrieve_top_memory(query="test")

    memory_update_args = {"access_time": str(get_epoch_time())}
    memory_store.update_memory_access_time(custom_id="573c58f2e90564478b301fe78c513c78")
