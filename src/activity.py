# Activity class should have parameters tool_list, activity_description, activity_input, overall_goal
## activity_input is the LLM generated subject of this activity
## overall_goal is the days overall objective, across all activities

# Generic activity consists of the following stages:
## 1. Plan subtasks. LLM call to generate 3 distinct questions related to the overall_goal and activity_input. Also passed identity statement and recent memories.
## 2. Subtasks 1-3. Each of these is a multi-step agent-executor followed by a constitutional chain at the end with the identity statement. The subtask should be passed identity statement, activity description, overall goal, recent memories.
## 3. Summarize activity. Given overall_goal, activity_input, and the list of 3x(subquestion+final answer), summarize your conclusions.
## 4. Reflect on what (3) means for the agent itself and the world.

import sys
import re
from typing import Any, List, Optional, Type, Dict

from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.chains.constitutional_ai.base import ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple
from langchain.prompts.prompt import PromptTemplate

sys.path.append("./")
from src.models import LlamaModelHandler, MistralModelHandler, EmbeddingHandler
from src.docs import DocumentHandler

# from src.tools import ToolHandler
from src.util import get_secrets, get_word_match_list, agent_logs
from src.memory_store import PGMemoryStoreSetter, PGMemoryStoreRetriever
from src.docs import AggregateRetrieval
from src.reflection import Reflection
from src.prompts.activity import *


class Activity:
    def __init__(
        self,
        overall_goal,
        activity_type,
        activity_input,
        memory_store_setter,
        memory_store_retriever,
        llm,
    ):
        self.overall_goal = overall_goal
        self.activity_type = activity_type
        self.activity_input = activity_input
        self.memory_store_setter = memory_store_setter
        self.memory_store_retriever = memory_store_retriever
        self.llm = llm

    def set_activity_toollist(self):
        pass

    def plan_subtasks(self):
        """Create a list of distinct subtasks based on the original activity input."""
        memory_kwargs = {
            "mem_to_search": 100,
            "mem_to_return": 4,
            "relevance_wt": 0,
            "importance_wt": 0.5,
            "recency_wt": 4,
            "update_access_time": False,
            "mem_type": "reflection",  # gets reflections only
        }

        recent_memories = AggregateRetrieval(
            vectorstore_retriever=self.memory_store_retriever
        ).run("", **memory_kwargs)

        subtask_plan = "1. " + self.llm(
            PLAN_SUBTASK_PROMPT.replace("{recent_memories}", recent_memories)
            .replace("{overall_goal}", self.overall_goal)
            .replace("{activity_type}", self.activity_type)
            .replace("{activity_input}", self.activity_input)
        )
        return subtask_plan

    def run_subtask(self):
        pass

    def summarize_activity(self):
        pass

    def reflect_activity(self):
        pass


if __name__ == "__main__":
    # model_name = "llama-13b"
    # lora_name = "alpaca-gpt4-lora-13b-3ep"
    # model_name = "llama-7b"
    # lora_name = "alpaca-lora-7b"
    testAgent = MistralModelHandler()
    eb = EmbeddingHandler().get_hf_embedding()

    pipeline, model, tokenizer = testAgent.load_mistral_llm()

    # Load memory store setter and retriever
    memory_store_setter = PGMemoryStoreSetter(embedding=eb)
    memory_store_retriever = PGMemoryStoreRetriever(embedding=eb)

    # load reflection
    reflection = Reflection(
        memory_store_setter=memory_store_setter,
        memory_store_retriever=memory_store_retriever,
        llm=pipeline,
    )

    # initiate activity
    activity = Activity(
        overall_goal="explore how writing helps us understand our own thoughts and feelings better as well as how it can be used to communicate effectively, develop critical thinking skills, and express ourselves.",
        activity_type="Research",
        activity_input="How does writing help us express ourselves?",
        memory_store_setter=memory_store_setter,
        memory_store_retriever=memory_store_retriever,
        llm=pipeline,
    )

    subtask_plan = activity.plan_subtasks()
