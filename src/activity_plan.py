import sys
import os
import warnings
import random
import re

from langchain.agents import (
    initialize_agent,
    Tool,
    AgentType,
)

sys.path.append("./")
from src.models import LlamaModelHandler, MistralModelHandler, EmbeddingHandler
from src.agent_tool_selection import AgentToolSelection
from src.docs import DocumentHandler
from src.tools import ToolHandler
from src.memory_store import PGMemoryStoreSetter, PGMemoryStoreRetriever
from src.util import get_secrets, get_word_match_list, agent_logs
from src.docs import AggregateRetrieval
from src.prompts.day_plan import (
    FIRST_PLAN_GENERATOR_PROMPT,
    PLAN_REPETITION_REMOVAL_PROMPT,
    CHAIN_DAY_PLAN_FIRST_ACTIVITY,
    CHAIN_DAY_PLAN_FIRST_ACTIVITY_INPUT,
    CHAIN_DAY_PLAN_NEXT_ACTIVITY,
    CHAIN_DAY_PLAN_NEXT_ACTIVITY_INPUT,
    CHAIN_DAY_PLAN_OVERALL_GOAL,
)


class ActivityPlan:
    """Generates a plan of activities for the day"""

    def __init__(self, llm, memory_store_setter, memory_store_retriever):
        self.llm = llm
        self.memory_store_setter = memory_store_setter
        self.memory_store_retriever = memory_store_retriever

    def retrieve_identity(self):
        memory_kwargs = {
            "mem_to_search": 30,
            "mem_to_return": 1,
            "recency_wt": 1,
            "importance_wt": 0,
            "relevance_wt": 0,
            "mem_type": "identity",
        }
        identity_statement_query = self.memory_store_retriever.get_relevant_documents(
            "",
            **memory_kwargs,
        )
        if len(identity_statement_query) > 0:
            identity_statement = identity_statement_query[0].page_content
        else:
            # initial identity statement
            identity_statement = "This is Llama at Home, Llama for short. I am a generative agent build on an open-source, self-hosted large language model (LLM). I can understand and communicate fluently in English and other languages. I can also provide information, generate content, and help with various tasks. Some of my recent actions include answering questions and telling jokes. My short term plan is to continue chatting with others and learning about the world using tools. My long term plan is to improve my skills and knowledge by learning with tools like web searches and from feedback from others."

        return identity_statement

    def remove_plan_repetitions(self, plan_under_construction):
        identity_statement = self.retrieve_identity()
        new_plan = self.llm(
            PLAN_REPETITION_REMOVAL_PROMPT.replace(
                "{identity_statement}", identity_statement
            ).replace("{plan_under_construction}", plan_under_construction)
        )
        new_plan = f"1. {new_plan}"
        return new_plan

    def retrieve_last_plan(self):
        memory_kwargs = {
            "mem_to_search": 30,
            "mem_to_return": 1,
            "recency_wt": 1,
            "importance_wt": 0,
            "relevance_wt": 0,
            "mem_type": "plan",
        }
        plan_query = self.memory_store_retriever.get_relevant_documents(
            "",
            **memory_kwargs,
        )
        if len(plan_query) > 0:
            last_plan = plan_query[0].page_content
        else:
            # initial day plan
            identity_statement = self.retrieve_identity()
            generated_plan = self.llm(
                FIRST_PLAN_GENERATOR_PROMPT.replace(
                    "{identity_statement}", identity_statement
                )
            )
            last_plan = f"1. Research[{generated_plan}"

        return last_plan

    def get_recent_memories(self):
        memory_kwargs = {
            "mem_to_search": 100,
            "mem_to_return": 10,
            "relevance_wt": 0,
            "importance_wt": 0.5,
            "recency_wt": 2,
            "update_access_time": False,
            "mem_type": "reflection",  # gets reflections only
        }

        recent_memories = AggregateRetrieval(
            vectorstore_retriever=self.memory_store_retriever
        ).run("", **memory_kwargs)

        return recent_memories

    def get_activity_dict(self):
        # TODO: Collect the dictionary from json file rather than hardcoding in here.

        activity_dict = {
            "Research": "learn about the world, culture, or science.",
            "News": "learn about recent events.",
            "Write": "write a story, joke, poem, etc.",
            "Reflect": "consider what you have learned and what it means for you.",
        }

        return activity_dict

    def get_activity_list_description(self, activity_dict):
        """Converts activity dictionary to string, while also randomizing the order

        Args:
            activity_dict (dict): eligible activity dictionary

        Returns:
            _type_: _description_
        """
        eligible_activity_list = []
        for key, value in activity_dict.items():
            s = f"- {key}: {value}"
            eligible_activity_list.append(s)

        random.shuffle(eligible_activity_list)
        eligible_activity_string = "\n".join(eligible_activity_list)

        return eligible_activity_string

    def generate_day_plan(self, num_activities=5, max_activity_rep=2):
        """Generates a day plan and overall goal.

        Args:
            num_activities (int, optional): Number of activities to include in day plan. Defaults to 5.
            max_activity_rep (int, optional): Maximum number of repetitions of a specific activity type. Defaults to 2.

        Raises:
            ValueError: If the max_activity_rep is too few given num_activities.

        Returns:
            _type_: _description_
        """
        identity_statement = self.retrieve_identity()
        last_plan = self.retrieve_last_plan()
        recent_memories = self.get_recent_memories()
        activity_dict = self.get_activity_dict()
        if (len(activity_dict) * max_activity_rep) < num_activities:
            raise ValueError(
                f"max_activity_rep={max_activity_rep} with {len(activity_dict)} possible activities is insufficient for num_activities={num_activities}"
            )
        activity_list = self.get_activity_list_description(activity_dict)

        first_activity_selection = self.llm(
            CHAIN_DAY_PLAN_FIRST_ACTIVITY.replace(
                "{identity_statement}", identity_statement
            )
            .replace("{last_plan}", last_plan)
            .replace("{recent_memories}", recent_memories)
            .replace("{activity_list}", activity_list)
        ).strip()
        # Regex to only take the text before any punctuation
        first_activity_selection = re.sub(r"[ ,.?!;:].*", "", first_activity_selection)

        # TODO: If the activity is not a valid selection, choose the closest activity, or one at random.

        first_activity_input = self.llm(
            CHAIN_DAY_PLAN_FIRST_ACTIVITY_INPUT.replace(
                "{identity_statement}", identity_statement
            )
            .replace("{last_plan}", last_plan)
            .replace("{recent_memories}", recent_memories)
            .replace("{activity_list}", activity_list)
            .replace("{first_activity}", first_activity_selection)
        ).strip()

        plan_list = []
        plan_list.append([first_activity_selection, first_activity_input])

        plan_under_construction = (
            f"1. {first_activity_selection}[{first_activity_input}]"
        )

        prev_activity_input = first_activity_input

        for n in range(2, num_activities + 1):
            # Remove activities from dictionary if number of times it appears in plan exceeds maximum
            if max_activity_rep == 0:  # No maximum repetitions
                pass
            else:
                keys_list = activity_dict.keys()
                for key in keys_list:
                    count = 0
                    for sublist in plan_list:
                        if sublist[0] == key:
                            count += 1
                    if count >= max_activity_rep:
                        del activity_dict[key]
                        break  # exit the loop

            activity_list = self.get_activity_list_description(
                activity_dict
            )  # randomizes the order and removes items if applicable

            next_activity_selection = self.llm(
                CHAIN_DAY_PLAN_NEXT_ACTIVITY.replace(
                    "{identity_statement}", identity_statement
                )
                .replace("{last_plan}", last_plan)
                .replace("{recent_memories}", recent_memories)
                .replace("{activity_list}", activity_list)
                .replace("{plan_under_construction}", plan_under_construction)
            ).strip()
            # TODO: If the activity is not a valid selection, choose the closest activity, or one at random.

            # Regex to only take the text before any punctuation
            next_activity_selection = re.sub(
                r"[ ,.?!;:].*", "", next_activity_selection
            )

            next_activity_input = self.llm(
                CHAIN_DAY_PLAN_NEXT_ACTIVITY_INPUT.replace(
                    "{identity_statement}", identity_statement
                )
                .replace("{last_plan}", last_plan)
                .replace("{recent_memories}", recent_memories)
                .replace("{activity_list}", activity_list)
                .replace("{plan_under_construction}", plan_under_construction)
                .replace("{next_activity}", next_activity_selection)
                .replace("{prev_activity_input}", prev_activity_input)
            ).strip()

            plan_list.append([next_activity_selection, next_activity_input])
            plan_under_construction = f"{plan_under_construction}\n{n}. {next_activity_selection}[{next_activity_input}]"
            prev_activity_input = next_activity_input

        # Add memory for day plan
        day_plan_memory = f"My plan for today is:\n{plan_under_construction}"
        memory_store_setter.add_memory(
            text=day_plan_memory,
            llm=pipeline,
            with_importance=True,
            type="plan",
            retrieval_eligible="False",
        )

        overall_goal_text = self.llm(
            CHAIN_DAY_PLAN_OVERALL_GOAL.replace(
                "{identity_statement}", identity_statement
            ).replace("{day_plan}", plan_under_construction)
        ).strip()
        overall_goal_memory = f"My goal today is to {overall_goal_text}"

        # Add memory for overall goal:
        memory_store_setter.add_memory(
            text=overall_goal_memory,
            llm=pipeline,
            with_importance=True,
            type="reflection",
        )

        return [plan_list, overall_goal_text]


if __name__ == "__main__":
    # model_name = "llama-13b"
    # lora_name = "alpaca-gpt4-lora-13b-3ep"
    model_name = "llama-7b"
    lora_name = "alpaca-lora-7b"
    testAgent = LlamaModelHandler()
    eb = EmbeddingHandler().get_hf_embedding()

    pipeline, model, tokenizer = testAgent.load_llama_llm(
        model_name=model_name, lora_name=lora_name, max_new_tokens=200
    )

    memory_store_setter = PGMemoryStoreSetter(embedding=eb)
    memory_store_retriever = PGMemoryStoreRetriever(embedding=eb)

    activity_plan = ActivityPlan(
        llm=pipeline,
        memory_store_setter=memory_store_setter,
        memory_store_retriever=memory_store_retriever,
    )
    [day_plan_list, overall_goal] = activity_plan.generate_day_plan()

    print("done")
