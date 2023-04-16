import logging
import sys
import os
import warnings

sys.path.append("./")

from langchain.utilities import (
    WikipediaAPIWrapper,
    SerpAPIWrapper,
    GoogleSearchAPIWrapper,
    SearxSearchWrapper,
)
from langchain.chains import ConversationChain, RetrievalQA
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import BaseTool
from src.my_langchain_models import MyLangchainLlamaModelHandler
from src.my_langchain_docs import MyLangchainDocsHandler

# suppress warnings for demo
warnings.filterwarnings("ignore")
os.environ["PYDEVD_INTERRUPT_THREAD_TIMEOUT"] = "60"
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT "] = "60"


class MyLangchainAgentExecutorHandler:
    """a wrapper to make creating a langchain agent executor easier"""

    def __init__(self, hf, tool_names, doc_info=dict(), **kwarg):
        self.hf = hf
        self.wiki_api = None
        self.searx_search = None
        self.google_search = None
        self.serp_search = None
        self.doc_retrivers = dict()
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
        # add document retrivers to tools
        if len(doc_info) > 0:
            newAgent = MyLangchainLlamaModelHandler()
            embedding = newAgent.load_hf_embedding()
            newDocs = MyLangchainDocsHandler(
                embedding=embedding, redis_host=self._get_secrets("redis_host")
            )
            for index_name in list(doc_info.keys()):
                index_tool_name = doc_info[index_name]["tool_name"]
                index_descripton = doc_info[index_name]["description"]
                index_filepaths = doc_info[index_name]["files"]
                index = newDocs.load_docs_into_redis(index_filepaths, index_name)
                vectorstore_retriever = index.vectorstore.as_retriever()
                self.doc_retrivers[index_name] = RetrievalQA.from_chain_type(
                    llm=hf, chain_type="stuff", retriever=vectorstore_retriever
                )
                tools.append(
                    Tool(
                        name=index_tool_name,
                        func=self.doc_retrivers[index_name].run,
                        description=index_descripton,
                    )
                )
            del newAgent, embedding
        # finalize agent initiation
        self.agent = initialize_agent(
            tools, self.hf, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )

        # set up log
        # targets = logging.StreamHandler(sys.stdout), logging.FileHandler("test.log")
        # logging.basicConfig(format="%(message)s", level=logging.INFO, handlers=targets)
        # log = open("test.log", "a")
        # sys.stdout = log
        # print = log.info()

    def run(self, main_prompt):
        # run agent executor chain
        display_header = (
            "\x1b[1;32m" + f"""\n\nQuestion: {main_prompt}\nThought:""" + "\x1b[0m"
        )
        print(display_header)
        return self.agent.run(main_prompt)

    def _get_secrets(self, key_name):
        _key_file = open(f"secrets/{key_name}.key", "r", encoding="utf-8")
        _key_value = _key_file.read()
        return _key_value

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
        summarized_text = self.hf(summarize_prompt)
        return summarized_text

    def _init_searx_search(self):
        # Searx API
        # https://python.langchain.com/en/latest/modules/agents/tools/examples/searx_search.html
        self.searx_search = SearxSearchWrapper(
            searx_host=self._get_secrets("searx_host"), k=3, engines=["google"]
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
        os.environ["GOOGLE_API_KEY"] = self._get_secrets("google2api")
        os.environ["GOOGLE_CSE_ID"] = self._get_secrets("google2cse")
        self.google_search = GoogleSearchAPIWrapper(k=3)  # top k results only
        google_tool = Tool(
            name="Google",
            func=self.google_search.run,
            description="recent events and specific facts about a topic.",
        )
        return google_tool

    def _init_serpapi(self):
        # Serp API
        os.environ["SERPAPI_API_KEY"] = self._get_secrets("serpapi")
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

    testAgent = MyLangchainLlamaModelHandler()
    eb = testAgent.load_hf_embedding()
    pipeline, model, tokenizer = testAgent.load_llama_llm(
        model_name=model_name, lora_name=lora_name, max_new_tokens=200
    )

    # define tool list (excluding any documents)
    test_tool_list = ["wiki", "searx"]

    # define test documents
    test_doc_info = {
        "examples": {
            "tool_name": "state_of_union",
            "description": "specific facts from the 2023 state of the union on Joe Biden's plan to rebuild the economy and unite the nation.",
            "files": ["index-docs/examples/state_of_the_union.txt"],
        },
        # "arxiv": {
        #     "tool_name": "arxiv",
        #     "description": "scientific papers from arxiv on math, science, and computer science.",
        #     "files": ["index-docs/arxiv/2302.13971.pdf"],
        # },
        "translink": {
            "tool_name": "translink",
            "description": "published policy documents on transportation in Metro Vancouver by TransLink.",
            "files": [
                "index-docs/translink/2020-11-12_capstan_open-house_boards.pdf",
                "index-docs/translink/2020-11-30_capstan-station_engagement-summary-report-final.pdf",
                "index-docs/translink/rail_to_ubc_rapid_transit_study_jan_2019.pdf",
                "index-docs/translink/t2050_10yr-priorities.pdf",
                "index-docs/translink/TransLink - Transport 2050 Regional Transportation Strategy.pdf",
                "index-docs/translink/translink-ubcx-summary-report-oct-2021.pdf",
                "index-docs/translink/ubc_line_rapid_transit_study_phase_2_alternatives_evaluation.pdf",
                "index-docs/translink/ubc_rapid_transit_study_alternatives_analysis_findings.pdf",
            ],
        },
    }

    # initiate agent executor
    test_agent_executor = MyLangchainAgentExecutorHandler(
        hf=pipeline, tool_names=test_tool_list, doc_info=test_doc_info
    )

    # testing start
    print("testing for agent executor starts...")
    # main_prompt = """What is the current inflation rate in the United States?"""
    # main_prompt = """Who leaked the document on Ukraine?"""
    # main_prompt = """Which city will be hosting the summer olympics in 2036?"""
    main_prompt = """Which city will be hosting the summer olympics in 2032?"""
    # main_prompt = """What is the current progress on nuclear fusion?"""
    # main_prompt = """What wars are happening around the world right now?"""
    # main_prompt = """Summarize the current events today on the US Economy."""
    # main_prompt = """What is the current financial situation of TransLink?"""
    test_agent_executor.run(main_prompt)

    # finish
    print("testing complete")
