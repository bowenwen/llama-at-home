import sys
import os

from langchain.utilities import (
    SearxSearchWrapper,
    WikipediaAPIWrapper,
    SerpAPIWrapper,
    GoogleSearchAPIWrapper,
)
from langchain.agents import Tool

sys.path.append("./")
from src.util import get_secrets, get_word_match_list, agent_logs


class ToolHandler:
    """a wrapper to make creating tools easier"""

    def __init__(self):
        pass

    def get_tools(self, tool_names, pipeline):
        tools = []
        if "wiki" in tool_names:
            tools.append(self._init_wiki_api(pipeline=pipeline))
        if "searx" in tool_names:
            tools.append(self._init_searx_search())
        if "google" in tool_names:
            tools.append(self._init_googleapi())
        if "serp" in tool_names and "google" not in tool_names:
            tools.append(self._init_serpapi())
        return tools

    @staticmethod
    def get_tools_list_descriptions(tools):
        return "\n".join([f"- {i.name}: {i.description}" for i in tools])

    @staticmethod
    def get_tools_list(tools):
        return [i.name for i in tools]

    def _init_wiki_api(self, pipeline):
        self.pipeline = pipeline
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
        summarize_prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction: Please provide a detailed summary of the following information \n### Input:\n{full_string} \n### Response: "
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

    def _init_googleapi(self, k=3):
        # Google API
        os.environ["GOOGLE_API_KEY"] = get_secrets("google2api")
        os.environ["GOOGLE_CSE_ID"] = get_secrets("google2cse")
        google_tool = Tool(
            name="Google",
            func=GoogleSearchAPIWrapper(k=3).run,
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
