import sys
import os
import warnings

from langchain.agents import (
    initialize_agent,
    Tool,
    AgentType,
)

sys.path.append("./")
from src.models import LlamaModelHandler
from src.agent_tool_selection import AgentToolSelection
from src.docs import DocumentHandler
from src.tools import ToolHandler
from src.memory_store import MemoryStore
from src.util import get_secrets, get_word_match_list, agent_logs
from src.prompt import TOOL_SELECTION_PROMPT

# suppress warnings for demo
warnings.filterwarnings("ignore")
os.environ["PYDEVD_INTERRUPT_THREAD_TIMEOUT"] = "60"
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT "] = "60"


class AgentExecutorHandler:
    """a wrapper to make creating a langchain agent executor easier"""

    def __init__(
        self,
        pipeline,
        embedding,
        tool_names,
        doc_info=dict(),
        run_tool_selector=False,
        update_long_term_memory=False,
        use_long_term_memory=False,
        **kwarg,
    ):
        self.kwarg = kwarg
        self.pipeline = pipeline
        self.embedding = embedding
        self.new_session = kwarg["new_session"] if "new_session" in kwarg else False
        self.use_cache_from_log = (
            kwarg["use_cache_from_log"] if "use_cache_from_log" in kwarg else False
        )
        self.update_long_term_memory = update_long_term_memory
        self.use_long_term_memory = use_long_term_memory
        self.long_term_memory_collection = (
            kwarg["long_term_memory_collection"]
            if "long_term_memory_collection" in kwarg
            else "long_term"
        )
        self.run_tool_selector = run_tool_selector
        self.log_tool_selector = (
            kwarg["log_tool_selector"] if "log_tool_selector" in kwarg else True
        )
        doc_use_type = kwarg["doc_use_type"] if "doc_use_type" in kwarg else "stuff"
        doc_top_k_results = (
            kwarg["doc_top_k_results"] if "doc_top_k_results" in kwarg else 3
        )
        # build tools
        tools_wrapper = ToolHandler()
        tools = tools_wrapper.get_tools(tool_names, pipeline)
        # add document retrievers to tools
        if len(doc_info) > 0:
            newDocs = DocumentHandler(
                embedding=embedding, redis_host=get_secrets("redis_host")
            )
            doc_tools = newDocs.get_tool_from_doc(
                pipeline=pipeline,
                doc_info=doc_info,
                doc_use_type=doc_use_type,
                doc_top_k_results=doc_top_k_results,
            )
            tools = tools + doc_tools
        # initialize memory bank
        if self.update_long_term_memory or self.use_long_term_memory:
            memory_tool = self._init_long_term_memory(embedding)
            if self.use_long_term_memory:
                tools.append(memory_tool)
        # finalize agent initiation
        self.agent = initialize_agent(
            tools,
            self.pipeline,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    def run(self, main_prompt):
        if self.new_session:
            # set cache state to save cache logs
            cached_response = agent_logs.set_cache_lookup(
                f"Agent Executor - {main_prompt}"
            )
            # if using cache from logs saved, then try to load previous log
            if cached_response is not None and self.use_cache_from_log:
                return cached_response
            agent_logs().clear_log()

        # initiate agent executor chain
        if self.run_tool_selector:
            selected_tools = AgentToolSelection(
                pipeline=self.pipeline,
                tools=self.agent.tools,
                **self.kwarg,
            )

            agent = initialize_agent(
                selected_tools,
                self.pipeline,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
            )
        else:
            agent = self.agent

        # print shortlist of tools being used
        tool_list_display = f"Tools available: {[i.name for i in agent.tools]}"
        print(tool_list_display)
        agent_logs.write_log(tool_list_display)
        # print question
        display_header = (
            "\x1b[1;32m" + f"""\n\nQuestion: {main_prompt}\nThought:""" + "\x1b[0m"
        )
        print(display_header)
        agent_logs.write_log(display_header)

        # run agent
        result = agent.run(main_prompt)

        if self.update_long_term_memory:
            self.memory_bank.add_memory(text=result, llm=self.pipeline)

        # always cache the current log
        agent_logs.save_cache()

        return result

    def _init_long_term_memory(self, embedding):
        self.memory_bank = MemoryStore(embedding, self.long_term_memory_collection)
        memory_tool = Tool(
            name="Memory",
            func=self.memory_bank.retrieve_memory_by_relevance_rank,
            description="knowledge bank based on previous conversations",
        )
        return memory_tool


if __name__ == "__main__":
    # test this class

    # select model and lora
    model_name = "llama-7b"
    lora_name = "alpaca-lora-7b"

    testAgent = LlamaModelHandler()
    eb = testAgent.get_hf_embedding()
    pipeline, model, tokenizer = testAgent.load_llama_llm(
        model_name=model_name, lora_name=lora_name, max_new_tokens=200
    )

    # define tool list (excluding any documents)
    # test_tool_list = ["wiki", "searx"]
    test_tool_list = []

    # define test documents
    test_doc_info = {
        "examples": {
            "tool_name": "State of Union QA system",
            "description": "specific facts from the 2023 state of the union on Joe Biden's plan to rebuild the economy and unite the nation.",
            "files": ["index-docs/examples/state_of_the_union.txt"],
        }
    }

    # initiate agent executor
    kwarg = {"doc_use_type": "stuff", "doc_top_k_results": 3}
    test_agent_executor = AgentExecutorHandler(
        pipeline=pipeline,
        embedding=eb,
        tool_names=test_tool_list,
        doc_info=test_doc_info,
        **kwarg,
    )

    # testing start
    print("testing for agent executor starts...")
    test_prompt = "What did the president say about Ketanji Brown Jackson"

    test_agent_executor.run(test_prompt)

    # finish
    print("testing complete")
