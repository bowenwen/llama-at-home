import sys
import re
from typing import Any, List, Optional, Type, Dict

from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate

sys.path.append("./")
from src.my_langchain_models import MyLangchainLlamaModelHandler
import src.prompt as prompts


class ChainSequence:
    """a simple wrapper around different LLM Chain types to perform complex task"""

    # https://python.langchain.com/en/latest/modules/chains/getting_started.html#create-a-custom-chain-with-the-chain-class

    def __init__(self, config, pipeline):
        """
        example for chains:
        chain_config = [
            {
                "name": "task1",
                "type": "simple",
                "input_template": "Give me one name for a company that makes {input}?",
            },
            {
                "name": "task2",
                "type": "simple",
                "input_template": "What is a good slogan for a company that makes {input} and named {task1_output}?",
            },
        ]
        """
        self.chains = dict()
        self.outputs = {"input": ""}
        # TODO: add support for chain serialization - https://python.langchain.com/en/latest/modules/chains/generic/serialization.html
        # notes on different chain types
        # https://python.langchain.com/en/latest/modules/chains/index_examples/summarize.html

        for c in config:
            task_name = c["name"]
            self.chains[task_name] = c
            if c["type"] == "simple":
                self.chains[task_name]["chain"] = self._init_llmchain(c, pipeline)
            else:
                raise ValueError("chain type not currently supported.")

    def run(self, input):
        print("> Initiating custom chain sequence...")
        self.outputs["input"] = input
        for task_name in list(self.chains.keys()):
            c = self.chains[task_name]
            values = [self.outputs[v] for v in c["input_vars"]]
            input_list = {key: value for key, value in zip(c["input_vars"], values)}
            self.outputs[f"{task_name}_output"] = (
                c["chain"].apply([input_list])[0]["text"].strip()
            )
            print(self.outputs[f"{task_name}_output"])

    def _init_llmchain(self, config, llm):
        template = config["input_template"]
        inputs = re.findall(r"\{(\w+)\}", template)
        config["input_vars"] = inputs
        prompt_template = PromptTemplate(
            input_variables=inputs,
            template=template,
        )
        chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
        return chain


if __name__ == "__main__":
    # test this class

    # select model and lora
    model_name = "llama-13b"
    lora_name = "alpaca-gpt4-lora-13b-3ep"

    testAgent = MyLangchainLlamaModelHandler()
    eb = testAgent.get_hf_embedding()
    pipeline, model, tokenizer = testAgent.load_llama_llm(
        model_name=model_name, lora_name=lora_name, max_new_tokens=200
    )

    chain_config = [
        {
            "name": "task1",
            "type": "simple",
            "input_template": prompts.CHAIN_EXAMPLE_1,
        },
        {
            "name": "task2",
            "type": "simple",
            "input_template": prompts.CHAIN_EXAMPLE_2,
        },
        {
            "name": "task3",
            "type": "simple",
            "input_template": prompts.CHAIN_EXAMPLE_3,
        },
    ]

    custom_chains = ChainSequence(config=chain_config, pipeline=pipeline)
    custom_chains.run(input="socks")
