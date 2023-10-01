import sys
import os
import warnings

from langchain.utilities import (
    SearxSearchWrapper,
    WikipediaAPIWrapper,
    SerpAPIWrapper,
    GoogleSearchAPIWrapper,
)
from langchain.agents import (
    initialize_agent,
    Tool,
    AgentType,
)

sys.path.append("./")
from src.models import LlamaModelHandler, MistralModelHandler, EmbeddingHandler
from src.docs import DocumentHandler
from src.tools import ToolHandler
from src.util import get_secrets, get_word_match_list, agent_logs
from src.prompts.tool_select import TOOL_SELECTION_PROMPT

# suppress warnings for demo
warnings.filterwarnings("ignore")
os.environ["PYDEVD_INTERRUPT_THREAD_TIMEOUT"] = "60"
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT "] = "60"


class AgentToolSelection:
    """a wrapper to make creating a langchain agent executor easier"""

    def __init__(
        self,
        pipeline,
        tools,
        **kwarg,
    ):
        self.tools = tools
        self.pipeline = pipeline
        self.log_tool_selector = (
            kwarg["log_tool_selector"] if "log_tool_selector" in kwarg else True
        )

    def run(self, main_prompt):
        # initiate agent executor chain
        tool_name = [f"- {i.name}: " for i in self.tools]
        tool_description = [f"{i.description}\n" for i in self.tools]
        tool_list_prompt = "".join(
            [
                item
                for sublist in zip(
                    tool_name,
                    tool_description,
                )
                for item in sublist
            ]
        )
        tool_selection_prompt = TOOL_SELECTION_PROMPT.replace(
            "{main_prompt}", main_prompt
        ).replace("{tool_list_prompt}", tool_list_prompt)

        tool_selection_display_header = "\n> Initiating tool selection prompt..."
        if self.log_tool_selector:
            agent_logs.write_log_and_print(tool_selection_display_header)

        selection_output = self.pipeline(tool_selection_prompt)

        tool_selection_display_result = "\x1b[1;34m" + selection_output + "\x1b[0m\n"
        if self.log_tool_selector:
            agent_logs.write_log_and_print(tool_selection_display_result)

        bool_selection_output = get_word_match_list(selection_output, ["true", "false"])

        selected_tools = []
        if len(bool_selection_output) == len(tool_name):
            # response is ok, parse tool availability
            for i in range(0, len(tool_name)):
                if bool_selection_output[i] == "true":
                    selected_tools.append(self.tools[i])
        else:
            selected_tools = self.tools

        return selected_tools


if __name__ == "__main__":
    # test this class

    # select model and lora
    model_name = "llama-7b"
    lora_name = "alpaca-lora-7b"

    testAgent = LlamaModelHandler()
    embedding = EmbeddingHandler().get_hf_embedding()
    pipeline, model, tokenizer = testAgent.load_llama_llm(
        model_name=model_name, lora_name=lora_name, max_new_tokens=200
    )

    # define tool list (excluding any documents)
    test_tool_list = ["wiki", "searx"]

    # define test documents
    test_doc_info = {
        "examples": {
            "tool_name": "State of Union Document",
            "description": "President Joe Biden's 2023 state of the union address.",
            "files": ["index-docs/examples/state_of_the_union.txt"],
        }
    }

    # build tools
    tools_wrapper = ToolHandler()
    tools = tools_wrapper.get_tools(test_tool_list, pipeline)
    # add document retrievers to tools
    if len(test_doc_info) > 0:
        newDocs = DocumentHandler(
            embedding=embedding, redis_host=get_secrets("redis_host")
        )
        doc_tools = newDocs.get_tool_from_doc(
            pipeline=pipeline,
            doc_info=test_doc_info,
            doc_use_type="stuff",
            doc_top_k_results=3,
        )
        tools = tools + doc_tools

    # initiate agent executor
    kwarg = {"log_tool_selector": True}
    test_agent_executor = AgentToolSelection(
        pipeline=pipeline,
        tools=tools,
        **kwarg,
    )

    # testing start
    print("testing for agent executor starts...")
    test_prompt = "What did the president say about Ketanji Brown Jackson"

    test_agent_executor.run(test_prompt)

    # finish
    print("testing complete")
