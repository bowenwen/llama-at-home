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
from langchain.chains import RetrievalQA

sys.path.append("./")
from src.models import LlamaModelHandler
from src.docs import DocumentHandler
from src.docs import AggregateRetrieval
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
        self.pipeline = pipeline
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
        self.wiki_api = None
        self.searx_search = None
        self.google_search = None
        self.serp_search = None
        self.doc_retrievers = dict()
        doc_use_qachain = (
            kwarg["doc_use_qachain"] if "doc_use_qachain" in kwarg else True
        )
        doc_top_k_results = (
            kwarg["doc_top_k_results"] if "doc_top_k_results" in kwarg else 3
        )
        # build tools
        tools = []
        if "wiki" in tool_names:
            tools.append(self._init_wiki_api())
        if "searx" in tool_names:
            tools.append(self._init_searx_search())
        if "google" in tool_names:
            tools.append(self._init_googleapi())
        if "serp" in tool_names and "google" not in tool_names:
            tools.append(self._init_serpapi()())
        # add document retrievers to tools
        if len(doc_info) > 0:
            newDocs = DocumentHandler(
                embedding=embedding, redis_host=get_secrets("redis_host")
            )
            for index_name in list(doc_info.keys()):
                index_tool_name = doc_info[index_name]["tool_name"]
                index_descripton = doc_info[index_name]["description"]
                index_filepaths = doc_info[index_name]["files"]
                index = newDocs.load_docs_into_redis(index_filepaths, index_name)
                vectorstore_retriever = index.vectorstore.as_retriever(
                    search_kwargs={"k": doc_top_k_results}
                )
                if doc_use_qachain:
                    self.doc_retrievers[index_name] = RetrievalQA.from_chain_type(
                        llm=pipeline,
                        chain_type="stuff",
                        retriever=vectorstore_retriever,
                    ).run
                else:
                    self.doc_retrievers[index_name] = AggregateRetrieval(
                        index_name, vectorstore_retriever
                    ).run
                tools.append(
                    Tool(
                        name=index_tool_name,
                        func=self.doc_retrievers[index_name],
                        description=index_descripton,
                    )
                )
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
        # set cache state to save cache logs
        cached_response = agent_logs.set_cache_lookup(f"Agent Executor - {main_prompt}")
        # if using cache from logs saved, then try to load previous log
        if cached_response is not None and self.use_cache_from_log:
            return cached_response
        elif self.use_cache_from_log is False:
            agent_logs().clear_log()

        # initiate agent executor chain
        if self.run_tool_selector:
            tool_name = [f"- {i.name}: " for i in self.agent.tools]
            tool_description = [f"{i.description}\n" for i in self.agent.tools]
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
            print(tool_selection_display_header, end="")
            if self.log_tool_selector:
                agent_logs.write_log(tool_selection_display_header)

            selection_output = self.pipeline(tool_selection_prompt)

            tool_selection_display_result = (
                "\x1b[1;34m" + selection_output + "\x1b[0m\n"
            )
            print(tool_selection_display_result)
            if self.log_tool_selector:
                agent_logs.write_log(tool_selection_display_result)

            bool_selection_output = get_word_match_list(
                selection_output, ["true", "false"]
            )

            selected_tools = []
            if len(bool_selection_output) == len(tool_name):
                # response is ok, parse tool availability
                for i in range(0, len(tool_name)):
                    if bool_selection_output[i] == "true":
                        selected_tools.append(self.agent.tools[i])
            else:
                selected_tools = self.agent.tools

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
            self.memory_bank.add_memory(result)

        # always cache the current log
        agent_logs.save_cache()

        return result

    def _init_long_term_memory(self, embedding):
        self.memory_bank = MemoryStore(
            embedding, self.long_term_memory_collection
        )
        memory_tool = Tool(
            name="Memory",
            func=self.memory_bank.retrieve_memory,
            description="knowledge bank based on previous conversations",
        )
        return memory_tool

    def _init_wiki_api(self):
        self.wiki_api = WikipediaAPIWrapper(top_k_results=1)
        wiki_tool = Tool(
            name="Wikipedia",
            func=self.summarize_wikipedia,
            # func=truncate_wikipedia,
            description="general information and well-established facts on a topic.",
        )
        return wiki_tool

    def truncate_wikipedia(self, input_string):
        full_string = self.wiki_api.run(input_string)
        truncated_string = full_string[0:2000]
        return truncated_string

    def summarize_wikipedia(self, input_string):
        # use LLM to summarize the meaning of the Wikipedia text
        full_string = self.wiki_api.run(input_string)
        summarize_prompt = f"### Instruction: Please provide a detailed summary of the following information \n### Input:\n{full_string} \n### Response: "
        summarized_text = self.pipeline(summarize_prompt)
        return summarized_text

    def _init_searx_search(self):
        # Searx API
        # https://python.langchain.com/en/latest/modules/agents/tools/examples/searx_search.html
        self.searx_search = SearxSearchWrapper(
            searx_host=get_secrets("searx_host"), k=3, engines=["google"]
        )
        searx_tool = Tool(
            name="Search",
            func=self.searx_google_search,
            description="recent events and specific facts about a topic.",
        )
        return searx_tool

    def searx_google_search(self, input_string):
        # reference: https://searx.github.io/searx/admin/engines.html
        # https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/web_base.html
        truncated_string = ""
        while truncated_string == "":
            try:
                truncated_string = self.searx_search.run(input_string)
                truncated_string = truncated_string.replace("\n", " ")
            except:
                truncated_string = ""
        return truncated_string

    def _init_googleapi(self):
        # Google API
        os.environ["GOOGLE_API_KEY"] = get_secrets("google2api")
        os.environ["GOOGLE_CSE_ID"] = get_secrets("google2cse")
        self.google_search = GoogleSearchAPIWrapper(k=3)  # top k results only
        google_tool = Tool(
            name="Google",
            func=self.google_search.run,
            description="recent events and specific facts about a topic.",
        )
        return google_tool

    def _init_serpapi(self):
        # Serp API
        os.environ["SERPAPI_API_KEY"] = get_secrets("serpapi")
        self.serp_search = SerpAPIWrapper()
        serp_tool = Tool(
            name="Google",
            func=self.serp_search.run,
            description="recent events and specific facts about a topic.",
        )
        return serp_tool


if __name__ == "__main__":
    # test this class

    # select model and lora
    model_name = "llama-13b"
    lora_name = "alpaca-gpt4-lora-13b-3ep"

    testAgent = LlamaModelHandler()
    eb = testAgent.get_hf_embedding()
    pipeline, model, tokenizer = testAgent.load_llama_llm(
        model_name=model_name, lora_name=lora_name, max_new_tokens=200
    )

    # define tool list (excluding any documents)
    test_tool_list = ["wiki", "searx"]

    # define test documents
    test_doc_info = {
        "examples": {
            "tool_name": "State of Union QA system",
            "description": "specific facts from the 2023 state of the union on Joe Biden's plan to rebuild the economy and unite the nation.",
            "files": ["index-docs/examples/state_of_the_union.txt"],
        }
    }

    # initiate agent executor
    kwarg = {"doc_use_qachain": False, "doc_top_k_results": 3}
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
