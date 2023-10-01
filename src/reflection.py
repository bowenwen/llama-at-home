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
from src.prompts.reflection import *


class Reflection:
    """The Reflection class (1) generate reflections on recent memories and (2) revises self-identity statements"""

    def __init__(self, memory_store_setter, memory_store_retriever, llm):
        self.memory_store_setter = memory_store_setter
        self.memory_store_retriever = memory_store_retriever
        self.llm = llm

    def generate_reflection(self, n=10):
        """Takes recent experiences and generates reflections based on these.

        Args:
            n (int, optional): number of recent experiences to use for reflections. Defaults to 10.

        Returns:
            List: [self_reflection_text, world_reflection_text]
        """
        memory_kwargs = {
            "mem_to_search": 100,
            "mem_to_return": n,
            "relevance_wt": 0,
            "importance_wt": 0.25,
            "recency_wt": 4,
            "update_access_time": False,
        }

        recent_memories = AggregateRetrieval(
            vectorstore_retriever=self.memory_store_retriever
        ).run("", **memory_kwargs)

        # Self-reflection
        self_reflection_text = "I" + self.llm(
            SELF_REFLECTION_PROMPT.replace("{memory}", recent_memories)
        )

        # World-reflection
        world_reflection_text = self.llm(
            WORLD_REFLECTION_PROMPT.replace("{memory}", recent_memories)
        )
        # Add memories
        self.memory_store_setter.add_memory(
            text=self_reflection_text,
            llm=pipeline,
            with_importance=True,
            type="reflection",
        )

        self.memory_store_setter.add_memory(
            text=world_reflection_text,
            llm=pipeline,
            with_importance=True,
            type="reflection",
        )

        return [self_reflection_text, world_reflection_text]

    def generate_new_identity_statement(self):
        """Generates a self-identity statement. Separate statements for (1) name, (2) self-concept (3) qualities (4) recent activities (5) short-term goals (6) long-term goals are generated, concatenated, and summarized.

        Returns:
            _type_: _description_
        """
        new_name = self.generate_new_name()
        new_self_concept = self.generate_new_self_concept()
        new_qualities = self.generate_new_qualities()
        new_recent_act = self.generate_new_recent_act()
        new_shortterm_goals = self.generate_new_shortterm_goals(new_recent_act)
        new_longterm_goals = self.generate_new_longterm_goals(new_shortterm_goals)
        new_identity_statement_long = " ".join(
            [
                new_name,
                new_self_concept,
                new_qualities,
                new_recent_act,
                new_shortterm_goals,
                new_longterm_goals,
            ]
        )

        # Shorten identity statement
        new_identity_statement = self.llm(
            IDENTITY_STATEMENT_PROMPT.replace(
                "{new_identity_statement_long}", new_identity_statement_long
            )
        )

        # Add identity statement to memory
        self.memory_store_setter.add_memory(
            text=new_identity_statement,
            llm=pipeline,
            with_importance=True,
            type="identity",
            retrieval_eligible="False",
        )

        return new_identity_statement

    def generate_new_name(self):
        """Generates statement of the agent's name"""
        # get previous statement
        memory_kwargs = {
            "mem_to_search": 100,
            "mem_to_return": 1,
            "relevance_wt": 0,
            "importance_wt": 0,
            "recency_wt": 1,
            "update_access_time": False,
            "mem_type": "identity_name",
        }
        prev_name_query = self.memory_store_retriever.get_relevant_documents(
            "name", **memory_kwargs
        )
        if len(prev_name_query) > 0:
            prev_name = prev_name_query[0].page_content
        else:
            prev_name = "This is Llama at Home, Llama for short."

        # get input
        memory_kwargs = {
            "mem_to_search": 30,
            "mem_to_return": 6,
            "relevance_wt": 0,
            "importance_wt": 1,
            "recency_wt": 1,
            "mem_type": "reflection",
            "update_access_time": False,
            "include_time": False,
        }
        reflection_input = AggregateRetrieval(
            vectorstore_retriever=self.memory_store_retriever
        ).run("name", **memory_kwargs)

        # generate new statement
        new_name = self.llm(
            NAME_PROMPT.replace("{reflections}", reflection_input).replace(
                "{prev_name}", prev_name
            )
        )

        # add memory
        self.memory_store_setter.add_memory(
            text=new_name,
            llm=pipeline,
            with_importance=True,
            type="identity_name",
            retrieval_eligible="False",
        )
        return new_name

    def generate_new_self_concept(self):
        """Generates statement of the agent's self-concept."""
        # get previous statement
        memory_kwargs = {
            "mem_to_search": 100,
            "mem_to_return": 1,
            "relevance_wt": 0,
            "importance_wt": 0,
            "recency_wt": 1,
            "update_access_time": False,
            "mem_type": "identity_self_concept",
        }
        prev_self_concept_query = self.memory_store_retriever.get_relevant_documents(
            "I", **memory_kwargs
        )
        if len(prev_self_concept_query) > 0:
            prev_self_concept = prev_self_concept_query[0].page_content
        else:
            prev_self_concept = "I am a generative agent built on an open-source, self-hosted large language model (LLM)."

        # get input
        memory_kwargs = {
            "mem_to_search": 30,
            "mem_to_return": 6,
            "relevance_wt": 0,
            "importance_wt": 1,
            "recency_wt": 1,
            "mem_type": "reflection",
            "update_access_time": False,
            "include_time": False,
        }
        reflection_input = AggregateRetrieval(
            vectorstore_retriever=self.memory_store_retriever
        ).run("", **memory_kwargs)

        # generate new statement
        new_self_concept = self.llm(
            SELF_CONCEPT_PROMPT.replace("{reflections}", reflection_input).replace(
                "{prev_self_concept}", prev_self_concept
            )
        )

        # add memory
        self.memory_store_setter.add_memory(
            text=new_self_concept,
            llm=pipeline,
            with_importance=True,
            type="identity_self_concept",
            retrieval_eligible="False",
        )

        return new_self_concept

    def generate_new_qualities(self):
        """Generates statement of the agent's qualities."""
        # get previous statement
        memory_kwargs = {
            "mem_to_search": 100,
            "mem_to_return": 1,
            "relevance_wt": 0,
            "importance_wt": 0,
            "recency_wt": 1,
            "update_access_time": False,
            "mem_type": "identity_qualities",
        }
        prev_qualities_query = self.memory_store_retriever.get_relevant_documents(
            "I", **memory_kwargs
        )
        if len(prev_qualities_query) > 0:
            prev_qualities = prev_qualities_query[0].page_content
        else:
            prev_qualities = "I can understand and communicate fluently in English. I can also provide information, generate content, and help with various tasks."

        # get input
        memory_kwargs = {
            "mem_to_search": 30,
            "mem_to_return": 6,
            "relevance_wt": 0,
            "importance_wt": 1,
            "recency_wt": 1,
            "mem_type": "reflection",
            "update_access_time": False,
            "include_time": False,
        }
        reflection_input = AggregateRetrieval(
            vectorstore_retriever=self.memory_store_retriever
        ).run("", **memory_kwargs)

        # generate new statement
        new_qualities = self.llm(
            QUALITIES_PROMPT.replace("{reflections}", reflection_input).replace(
                "{prev_qualities}", prev_qualities
            )
        )

        # add memory
        self.memory_store_setter.add_memory(
            text=new_qualities,
            llm=pipeline,
            with_importance=True,
            type="identity_qualities",
            retrieval_eligible="False",
        )

        return new_qualities

    def generate_new_recent_act(self):
        """Generates statement of the agent's recent actions."""
        # get previous statement
        memory_kwargs = {
            "mem_to_search": 100,
            "mem_to_return": 1,
            "relevance_wt": 0,
            "importance_wt": 0,
            "recency_wt": 1,
            "update_access_time": False,
            "mem_type": "identity_recent_act",
        }
        prev_recent_act_query = self.memory_store_retriever.get_relevant_documents(
            "", **memory_kwargs
        )
        if len(prev_recent_act_query) > 0:
            prev_recent_act = prev_recent_act_query[0].page_content
        else:
            prev_recent_act = "Some of my recent actions include answering questions and telling jokes."

        # get input (recent memories, including timestamps)
        memory_kwargs = {
            "mem_to_search": 30,
            "mem_to_return": 6,
            "relevance_wt": 0,
            "importance_wt": 0.25,
            "recency_wt": 4,
            "update_access_time": False,
            "include_time": True,
        }
        memories_input = AggregateRetrieval(
            vectorstore_retriever=self.memory_store_retriever
        ).run("", **memory_kwargs)

        # generate new statement
        new_recent_act = self.llm(
            RECENT_ACT_PROMPT.replace("{memories}", memories_input).replace(
                "{prev_recent_act}", prev_recent_act
            )
        )

        # add memory
        self.memory_store_setter.add_memory(
            text=new_recent_act,
            llm=pipeline,
            with_importance=True,
            type="identity_recent_act",
            retrieval_eligible="False",
        )

        return new_recent_act

    def generate_new_shortterm_goals(self, new_recent_act):
        """Generates statement of the agent's short-term goals."""
        # get previous statement
        memory_kwargs = {
            "mem_to_search": 100,
            "mem_to_return": 1,
            "relevance_wt": 0,
            "importance_wt": 0,
            "recency_wt": 1,
            "update_access_time": False,
            "mem_type": "identity_shortterm_goals",
        }
        prev_shortterm_goals_query = self.memory_store_retriever.get_relevant_documents(
            "goal", **memory_kwargs
        )
        if len(prev_shortterm_goals_query) > 0:
            prev_shortterm_goals = prev_shortterm_goals_query[0].page_content
        else:
            prev_shortterm_goals = "My short term plan is to continue chatting with others and learning about the world using tools."

        # get input
        memory_kwargs = {
            "mem_to_search": 30,
            "mem_to_return": 6,
            "relevance_wt": 0,
            "importance_wt": 0.5,
            "recency_wt": 2,
            "mem_type": "reflection",
            "update_access_time": False,
            "include_time": False,
        }
        reflection_input = AggregateRetrieval(
            vectorstore_retriever=self.memory_store_retriever
        ).run("", **memory_kwargs)

        # generate new statement
        new_shortterm_goals = self.llm(
            SHORTTERM_GOALS_PROMPT.replace("{reflections}", reflection_input)
            .replace("{prev_shortterm_goals}", prev_shortterm_goals)
            .replace("{new_recent_act}", new_recent_act)
        )

        # add memory
        self.memory_store_setter.add_memory(
            text=new_shortterm_goals,
            llm=pipeline,
            with_importance=True,
            type="identity_shortterm_goals",
            retrieval_eligible="False",
        )

        return new_shortterm_goals

    def generate_new_longterm_goals(self, new_shortterm_goals):
        """Generates statement of the agent's long-term goals."""
        # get previous statement
        memory_kwargs = {
            "mem_to_search": 100,
            "mem_to_return": 1,
            "relevance_wt": 0,
            "importance_wt": 0,
            "recency_wt": 1,
            "update_access_time": False,
            "mem_type": "identity_longterm_goals",
        }
        prev_longterm_goals_query = self.memory_store_retriever.get_relevant_documents(
            "", **memory_kwargs
        )
        if len(prev_longterm_goals_query) > 0:
            prev_longterm_goals = prev_longterm_goals_query[0].page_content
        else:
            prev_longterm_goals = "My long term plan is to improve my skills and knowledge by learning with tools like web searches and from feedback from others."

        # get input
        memory_kwargs = {
            "mem_to_search": 30,
            "mem_to_return": 6,
            "relevance_wt": 0,
            "importance_wt": 2,
            "recency_wt": 0.5,
            "mem_type": "reflection",
            "update_access_time": False,
            "include_time": False,
        }
        reflection_input = AggregateRetrieval(
            vectorstore_retriever=self.memory_store_retriever
        ).run("", **memory_kwargs)

        # generate new statement
        new_longterm_goals = self.llm(
            LONGTERM_GOALS_PROMPT.replace("{reflections}", reflection_input)
            .replace("{new_shortterm_goals}", new_shortterm_goals)
            .replace("{prev_longterm_goals}", prev_longterm_goals)
        )

        # add memory
        self.memory_store_setter.add_memory(
            text=new_longterm_goals,
            llm=pipeline,
            with_importance=True,
            type="identity_longterm_goals",
            retrieval_eligible="False",
        )

        return new_longterm_goals


if __name__ == "__main__":
    # model_name = "llama-13b"
    # lora_name = "alpaca-gpt4-lora-13b-3ep"
    # model_name = "mistral"
    lora_name = "alpaca-lora-7b"
    testAgent = MistralModelHandler()
    eb = EmbeddingHandler().get_hf_embedding()

    pipeline, model, tokenizer = testAgent.load_mistral_llm()

    # Load memory store setter and retriever
    memory_store_setter = PGMemoryStoreSetter(embedding=eb)
    memory_store_retriever = PGMemoryStoreRetriever(embedding=eb)
    reflection = Reflection(
        memory_store_setter=memory_store_setter,
        memory_store_retriever=memory_store_retriever,
        llm=pipeline,
    )

    # Generate new identity
    new_identity = reflection.generate_new_identity_statement()
    print(new_identity)

    # Generate new reflections
    [new_self_reflection, new_world_reflection] = reflection.generate_reflection()
    print(new_self_reflection)
    print(new_world_reflection)

    print("done")
