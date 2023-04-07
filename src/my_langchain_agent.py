import sys

sys.path.append("./")

from pathlib import Path
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
