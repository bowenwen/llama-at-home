import sys

sys.path.append("./")

from pathlib import Path
from scipy.spatial import distance
from typing import Any, List, Optional, Type

from langchain.llms import HuggingFacePipeline
from transformers import (
    # AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from langchain.embeddings import HuggingFaceEmbeddings
from src.my_langchain_docs import MyLangchainDocsHandler


class MyLangchainAgentHandler:
    """a wrapper to make creating a langchain agent easier"""

    model_name = None
    max_new_tokens = None
    model = None
    tokenizer = None
    embedding = None
    hf = None
    model_loaded = False

    def __init__(self) -> None:
        """init function"""
        pass

    @classmethod
    def get_llama_llm(cls) -> Type[HuggingFacePipeline]:
        """get a previously loaded llama, if it was never loaded, return llama-7b with max_new_tokens=50

        Returns:
            model_asset_tuple: a size three tuple of HuggingFacePipeline, model and tokenizer
        """
        if cls.model_loaded is False:
            cls.load_llama_llm(model_name="llama-7b", max_new_tokens=50)
            return (cls.hf, cls.model, cls.tokenizer)
        return (cls.hf, cls.model, cls.tokenizer)

    @classmethod
    def load_llama_llm(
        cls, model_name=None, max_new_tokens=50
    ) -> Type[HuggingFacePipeline]:
        """quick loader of a local llama model

        Args:
            model_name (str, optional): the model name of llama or the folder name with models folder. Defaults to None.
            max_new_tokens (int, optional): max token size used for model. Defaults to 50.

        Returns:
            model_asset_tuple: a size three tuple of HuggingFacePipeline, model and tokenizer
        """
        if cls.model_loaded and cls.model_name == model_name:
            # return previously loaded model
            return (cls.hf, cls.model, cls.tokenizer)

        if model_name == None:
            cls.model_name = "llama-7b"
        else:
            cls.model_name = model_name

        model_path = f"models/{cls.model_name}"
        cls.max_new_tokens = max_new_tokens

        # TODO: review config.json with model folder:
        # https://huggingface.co/docs/transformers/v4.27.2/en/internal/generation_utils#transformers.TemperatureLogitsWarper

        cls.tokenizer = AutoTokenizer.from_pretrained(Path(f"{model_path}/"))
        cls.tokenizer.truncation_side = "left"
        cls.model = AutoModelForCausalLM.from_pretrained(
            Path(model_path),
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        # model = model.cuda()
        pipe = pipeline(
            "text-generation",
            model=cls.model,
            tokenizer=cls.tokenizer,
            max_new_tokens=cls.max_new_tokens,
        )
        cls.hf = HuggingFacePipeline(pipeline=pipe)
        cls.model_loaded = True
        return (cls.hf, cls.model, cls.tokenizer)

    @classmethod
    def get_hf_embedding(cls) -> Type[HuggingFaceEmbeddings]:
        if cls.embedding == None:
            return cls.load_hf_embedding()
        return cls.embedding

    @classmethod
    def load_hf_embedding(cls) -> Type[HuggingFaceEmbeddings]:
        """load default embedding used

        Returns:
            HuggingFaceEmbeddings: hugging face embedding model
        """

        # using the default embedding model from hf
        cls.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        return cls.embedding


if __name__ == "__main__":

    # test embeddings
    testAgent = MyLangchainAgentHandler()
    embedding = testAgent.load_hf_embedding()
    text1 = (
        "This is a very long sentence and the only difference is a period at the end"
    )
    text2 = (
        "This is a very long sentence and the only difference is a period at the end."
    )
    query_result1 = embedding.embed_query(text1)
    print(f"text1 result:\n{str(query_result1[0:5]).replace(']','...')}")
    query_result2 = embedding.embed_query(text2)
    print(f"text2 result:\n{str(query_result2[0:5]).replace(']','...')}")

    print(
        f"text1 and text2 distance: {distance.euclidean(query_result1, query_result2)}"
    )
    doc_result = embedding.embed_documents([text1, text2])
    print(f"text1 within doc:\n{str(doc_result[0][0:5]).replace(']','...')}")
    print(f"text2 within doc:\n{str(doc_result[1][0:5]).replace(']','...')}")

    # load llm
    hf, model, tokenizer = testAgent.load_llama_llm(
        model_name="llama-7b", max_new_tokens=50
    )

    # index documents
    index_name = "examples"
    file_list = [
        "docs/examples/state_of_the_union.txt",
        # "docs/arxiv/2302.13971.pdf",
        # "docs/psych/DSM-5-TR.pdf",
        # "docs/psych/Synopsis_of_Psychiatry.pdf",
    ]
    testDocs = MyLangchainDocsHandler(embedding=embedding)
    index = testDocs.load_docs_into_chroma(file_list, index_name)
    # index = testDocs.load_docs_into_redis(file_list, index_name)

    # using the VectorStoreIndexWrapper to run a chain with question related to the doc
    query = "What did the president say about Ketanji Brown Jackson"
    doc_response = index.query(query, llm=hf)
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
