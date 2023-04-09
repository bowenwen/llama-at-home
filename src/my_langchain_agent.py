from typing import Any, List, Optional, Type

# import torch

from pathlib import Path
from peft import PeftModelForCausalLM
from langchain.llms import HuggingFacePipeline
from transformers import (
    # AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from tokenizers import AddedToken
from langchain.embeddings import HuggingFaceEmbeddings

import sys

sys.path.append("./")

from langchain.callbacks import BaseCallbackHandler


class MyLangchainAgentHandler:
    """a wrapper to make creating a langchain agent easier"""

    model_name = ""
    lora_name = ""
    max_new_tokens = 50
    model = None
    tokenizer = None
    embedding = None
    hf = None
    model_loaded = False
    DIR_MODELS = "models"
    DIR_LORAS = "loras"
    pipeline_args = {
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 0.1,
        "typical_p": 1.0,
        "repetition_penalty": 1.18,
        "encoder_repetition_penalty": 1,
        "top_k": 40,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "penalty_alpha": 0,
        "length_penalty": 1,
        "early_stopping": False,
        "eos_token_id": [2],
        "stopping_criteria": [],
    }

    def __init__(self) -> None:
        """init function"""
        pass

    @classmethod
    def add_lora_to_model(cls, lora_name) -> None:
        # initialize default arguments
        if lora_name not in [None, "None", ""]:
            cls.lora_name = lora_name
        lora_dir = f"{cls.DIR_LORAS}/{lora_name}"
        cpu = 0
        load_in_8bit = 1

        # credit to text-generation-webui
        if lora_name not in ["None", ""]:
            print(f"Adding the LoRA {lora_name} to the model...")
            params = {}
            if not cpu:
                params["dtype"] = cls.model.dtype
                if hasattr(cls.model, "hf_device_map"):
                    params["device_map"] = {
                        "base_model.model." + k: v
                        for k, v in cls.model.hf_device_map.items()
                    }
                elif load_in_8bit:
                    params["device_map"] = {"": 0}

            cls.model = PeftModelForCausalLM.from_pretrained(
                cls.model, Path(f"{lora_dir}/"), **params
            )
            # NOT IMPLEMENTED
            # if not load_in_8bit and not cpu:
            #     cls.model.half()
            #     if not hasattr(cls.model, "hf_device_map"):
            #         if torch.has_mps:
            #             device = torch.device("mps")
            #             cls.model = cls.model.to(device)
            #         else:
            #             cls.model = cls.model.cuda()

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
        cls, model_name=None, lora_name=None, max_new_tokens=50
    ) -> Type[HuggingFacePipeline]:
        """quick loader of a local llama model

        Args:
            model_name (str, optional): the model name of llama or the folder name with models folder. Defaults to None.
            lora_name (str, optional): the name of lora peft fine tuning model. Defaults to None.
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

        model_path = f"{cls.DIR_MODELS}/{cls.model_name}"
        cls.max_new_tokens = max_new_tokens

        # TODO: review config.json with model folder:
        # https://huggingface.co/docs/transformers/v4.27.2/en/internal/generation_utils#transformers.TemperatureLogitsWarper

        # load model
        if "llama" in cls.model_name.lower():
            cls.model = LlamaForCausalLM.from_pretrained(
                Path(model_path),
                # low_cpu_mem_usage = True,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            )
        else:
            cls.model = AutoModelForCausalLM.from_pretrained(Path(model_path))
        # load tokenizer
        if type(cls.model) is LlamaForCausalLM:
            cls.tokenizer = LlamaTokenizer.from_pretrained(
                Path(f"{model_path}/"), clean_up_tokenization_spaces=True
            )
        else:
            cls.tokenizer = AutoTokenizer.from_pretrained(Path(f"{model_path}/"))
        cls.tokenizer.truncation_side = "left"
        # # fix missing new line characters in generation
        # cls.tokenizer.add_tokens(AddedToken("\n", normalized=False))
        # cls.tokenizer.add_tokens(AddedToken("\t", normalized=False))
        # cls.tokenizer.add_tokens(AddedToken("\n\t", normalized=False))
        # model = model.cuda()
        # load lora model if applicable
        if lora_name not in [None, "None", ""]:
            cls.add_lora_to_model(lora_name=lora_name)
        # build hf pipeline for langchain
        pipe = pipeline(
            model=cls.model,
            task="text-generation",
            framework="pt",
            tokenizer=cls.tokenizer,
            max_new_tokens=cls.max_new_tokens,
            **cls.pipeline_args,
        )
        # # fix missing new line characters for the model
        # cls.model.resize_token_embeddings(len(cls.tokenizer))
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
