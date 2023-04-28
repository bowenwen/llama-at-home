from typing import Any, List, Optional, Type

# import torch

from pathlib import Path
from peft import PeftModelForCausalLM
from transformers import (
    # AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
import torch
from tokenizers import AddedToken
from langchain.llms import HuggingFacePipeline
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks import BaseCallbackHandler

import sys

sys.path.append("./")

from src.GPTQ_loader import load_quantized


class LlamaModelHandler:
    """a wrapper to make creating a langchain agent easier"""

    model_name = ""
    lora_name = ""
    max_new_tokens = 50
    model = None
    tokenizer = None
    embedding = None
    hf = None
    device = None
    model_loaded = False
    quantized = False
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
    def load_gptq_quantized(cls, model_name: str):
        """a simplified version of loading gptq model for llama with 4 bits
        originally from https://github.com/oobabooga/text-generation-webui

        Args:
            model_name (str): name of the model

        Returns:
            AutoModelForCausalLM: model with adjustments made for 4 bit quantization
        """
        # prepare parameters
        kwargs_quant = {
            "wbits": 4,
            "groupsize": 128,
            "pre_layer": 0,  # number of layers to gpu, enables cpu offloading
        }
        # load model
        cls.model = load_quantized(model_name, **kwargs_quant)
        # torch.device("cuda:0")
        return cls.model

    @classmethod
    def get_llama_cpp_llm(cls) -> Type[LlamaCpp]:
        if cls.model_loaded is False:
            cls.load_llama_cpp_llm(model_name="llama-7b", max_new_tokens=200)
            return (cls.hf, cls.model, cls.tokenizer)
        return (cls.hf, cls.model, cls.tokenizer)

    @classmethod
    def load_llama_cpp_llm(
        cls,
        model_name=None,
        lora_name=None,
        context_window=1024,
        max_new_tokens=200,
        quantized=False,
    ) -> Type[LlamaCpp]:
        if cls.model_loaded and cls.model_name == model_name:
            # return previously loaded model
            return (cls.model, cls.tokenizer)

        if model_name == None:
            cls.model_name = "llama-7b"
        else:
            cls.model_name = model_name

        cls.quantized = quantized
        cls.max_new_tokens = max_new_tokens

        model_path = f"{cls.DIR_MODELS}/{cls.model_name}"
        lora_dir = f"{cls.DIR_LORAS}/{lora_name}"

        if lora_name not in [None, "None", ""] and cls.quantized:
            cls.model = LlamaCpp(
                model_path=f"{model_path}/ggml-model-q4_0.bin",
                lora_base=f"{model_path}/ggml-model-f16.bin",
                lora_path=f"{lora_dir}/ggml-adapter-model.bin",
                n_ctx=context_window,
                max_tokens=cls.max_new_tokens,
                temperature=cls.pipeline_args["temperature"],
                top_p=cls.pipeline_args["top_p"],
                echo=False,
                repeat_penalty=cls.pipeline_args["repetition_penalty"],
                top_k=cls.pipeline_args["top_k"],
            )
        elif lora_name not in [None, "None", ""] and not cls.quantized:
            cls.model = LlamaCpp(
                model_path=f"{model_path}/ggml-model-f16.bin",
                lora_path=f"{lora_dir}/ggml-adapter-model.bin",
                n_ctx=context_window,
                max_tokens=cls.max_new_tokens,
                temperature=cls.pipeline_args["temperature"],
                top_p=cls.pipeline_args["top_p"],
                echo=False,
                repeat_penalty=cls.pipeline_args["repetition_penalty"],
                top_k=cls.pipeline_args["top_k"],
            )
        elif cls.quantized:
            cls.model = LlamaCpp(
                model_path=f"{model_path}/ggml-model-q4_0.bin",
                n_ctx=context_window,
                max_tokens=cls.max_new_tokens,
                temperature=cls.pipeline_args["temperature"],
                top_p=cls.pipeline_args["top_p"],
                echo=False,
                repeat_penalty=cls.pipeline_args["repetition_penalty"],
                top_k=cls.pipeline_args["top_k"],
            )
        else:
            cls.model = LlamaCpp(
                model_path=f"{model_path}/ggml-model-f16.bin",
                n_ctx=context_window,
                max_tokens=cls.max_new_tokens,
                temperature=cls.pipeline_args["temperature"],
                top_p=cls.pipeline_args["top_p"],
                echo=False,
                repeat_penalty=cls.pipeline_args["repetition_penalty"],
                top_k=cls.pipeline_args["top_k"],
            )
        return cls.model

    # @classmethod
    # def get_llama_cpp_embedding(cls, model_name) -> Type[LlamaCppEmbeddings]:
    #     if cls.embedding == None:
    #         return cls.load_llama_cpp_embedding(model_name)
    #     return cls.embedding

    # @classmethod
    # def load_llama_cpp_embedding(cls, model_name) -> Type[LlamaCppEmbeddings]:
    #     """load default embedding used

    #     Returns:
    #         HuggingFaceEmbeddings: hugging face embedding model
    #     """
    #     cls.model_name = model_name
    #     model_path = f"{cls.DIR_MODELS}/{cls.model_name}"
    #     if cls.quantized:
    #         cls.embedding = LlamaCppEmbeddings(
    #             model_path=f"{model_path}/ggml-model-q4_0.bin"
    #         )
    #     else:
    #         cls.embedding = LlamaCppEmbeddings(
    #             model_path=f"{model_path}/ggml-model-f16.bin"
    #         )
    #     return cls.embedding

    @classmethod
    def get_llama_llm(cls) -> Type[HuggingFacePipeline]:
        """get a previously loaded llama, if it was never loaded, return llama-7b with max_new_tokens=200

        Returns:
            model_asset_tuple: a size three tuple of HuggingFacePipeline, model and tokenizer
        """
        if cls.model_loaded is False:
            cls.load_llama_llm(model_name="llama-7b", max_new_tokens=200)
            return (cls.hf, cls.model, cls.tokenizer)
        return (cls.hf, cls.model, cls.tokenizer)

    @classmethod
    def load_llama_llm(
        cls, model_name=None, lora_name=None, max_new_tokens=200
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
            # check if model is 4 bits
            if "4bit-128g" in cls.model_name.lower():
                lora_name = None
                print(
                    "Lora not implemented for 4bit-128g llama models. Setting lora_name to None"
                )
                cls.model = cls.load_gptq_quantized(model_name=model_name)
            else:
                cls.model = LlamaForCausalLM.from_pretrained(
                    Path(model_path),
                    # low_cpu_mem_usage = True,
                    device_map="auto",
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                )
        else:
            cls.model = AutoModelForCausalLM.from_pretrained(Path(model_path))
        # set device
        cls.device = cls.model.device
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
            device=cls.device,
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

        WARNING - if you change embedding, all previously calculated embedding must be recalculated

        source:
        - https://huggingface.co/models?library=sentence-transformers&sort=downloads
        - https://huggingface.co/spaces/mteb/leaderboard
        - https://huggingface.co/hkunlp/instructor-large
        - https://huggingface.co/sentence-transformers/all-mpnet-base-v2

        Returns:
            HuggingFaceEmbeddings: hugging face embedding model
        """

        # using the default embedding model from hf
        cls.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        return cls.embedding


if __name__ == "__main__":
    # test this class

    model_name = "llama-7b-4bit-128g"
    lora_name = "alpaca-lora-7b"

    testAgent = LlamaModelHandler()
    embedding = testAgent.get_hf_embedding()
    pipeline, model, tokenizer = testAgent.load_llama_llm(
        model_name=model_name, lora_name=lora_name, max_new_tokens=200
    )

    # test embedding
    text1 = (
        "This is a very long sentence and the only difference is a period at the end"
    )
    query_result1 = embedding.embed_query(text1)

    # test llm
    response = pipeline("hello")

    # finish
    print("testing complete")
