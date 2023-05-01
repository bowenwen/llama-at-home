import sys
import os

# from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI

sys.path.append("./")
from src.models import LlamaModelHandler
from src.agent_executor import AgentExecutorHandler
from src.gradio_ui import WebUI
from src.util import agent_logs, get_secrets
from src.chain_sequence import ChainSequence
import src.prompt as prompts

# run the current implementation of llama-at-home

# set environment variable to save chat history
os.environ["MYLANGCHAIN_SAVE_CHAT_HISTORY"] = "1"

# select model and lora
testAgent = LlamaModelHandler()
eb = testAgent.load_hf_embedding()

# define model used
# supported models: ["gpt-35-turbo", "llama-7b", "llama-13b", "llama-30b", "llama-65b"]
model_name = "llama-7b"

if model_name == "gpt-35-turbo":
    # https://python.langchain.com/en/latest/modules/models/llms/integrations/azure_openai_example.html
    # https://python.langchain.com/en/latest/modules/models/chat/integrations/azure_chat_openai.html
    pipeline = AzureChatOpenAI(
        openai_api_base=get_secrets("azure_openapi_url"),
        deployment_name=get_secrets("azure_openai_deployment"),
        openai_api_key=get_secrets("azure_openapi"),
        openai_api_type="azure",
        openai_api_version="2023-03-15-preview",
        model_name=model_name,
        temperature=0.1,
        max_tokens=200,
    )
elif model_name in ["llama-7b", "llama-13b"]:
    if model_name == "llama-7b":
        lora_name = "alpaca-lora-7b"
    if model_name == "llama-13b":
        lora_name = "alpaca-gpt4-lora-13b-3ep"
    pipeline, model, tokenizer = testAgent.load_llama_llm(
        model_name=model_name, lora_name=lora_name, max_new_tokens=200
    )
elif model_name in ["llama-30b", "llama-65b"]:
    if model_name == "llama-65b":
        lora_name = "alpaca-lora-65b-chansung"
    if model_name == "llama-30b":
        lora_name = "alpaca-lora-30b-chansung"
    pipeline = testAgent.load_llama_cpp_llm(
        model_name=model_name,
        lora_name=lora_name,
        context_window=8192,
        max_new_tokens=200,
        quantized=True,
    )
else:
    raise NotImplementedError(f"model name {model_name} not supported.")


# # define tool list (excluding any documents)
# test_tool_list = ["wiki", "searx"]
# # define test documents
# test_doc_info = {
#     # "examples": {
#     #     "tool_name": "State of Union QA system",
#     #     "description": "specific facts from the 2023 state of the union on Joe Biden's plan to rebuild the economy and unite the nation.",
#     #     "files": ["index-docs/examples/state_of_the_union.txt"],
#     # },
#     # "arxiv": {
#     #     "tool_name": "Arxiv Papers",
#     #     "description": "scientific papers from arxiv on math, science, and computer science.",
#     #     "files": [
#     #         "index-docs/arxiv/2302.13971.pdf",
#     #         "index-docs/arxiv/2304.03442.pdf",
#     #     ],
#     # },
#     "translink": {
#         "tool_name": "Translink Reports",
#         "description": "published policy documents on transportation in Metro Vancouver by TransLink.",
#         "files": [
#             "index-docs/translink/2020-11-12_capstan_open-house_boards.pdf",
#             "index-docs/translink/2020-11-30_capstan-station_engagement-summary-report-final.pdf",
#             "index-docs/translink/rail_to_ubc_rapid_transit_study_jan_2019.pdf",
#             "index-docs/translink/t2050_10yr-priorities.pdf",
#             "index-docs/translink/TransLink - Transport 2050 Regional Transportation Strategy.pdf",
#             "index-docs/translink/translink-ubcx-summary-report-oct-2021.pdf",
#             "index-docs/translink/ubc_line_rapid_transit_study_phase_2_alternatives_evaluation.pdf",
#             "index-docs/translink/ubc_rapid_transit_study_alternatives_analysis_findings.pdf",
#         ],
#     },
#     "psych_dsm": {
#         "tool_name": "Psychiatry DSM",
#         "description": "Diagnostic and Statistical Manual of Mental Disorders helps clinicians and researchers define and classify mental disorders, which can improve diagnoses, treatment, and research.",
#         "files": ["index-docs/psych/DSM-5-TR.pdf"],
#     },
#     # "psych_sop": {
#     #     "tool_name": "Synopsis of Psychiatry",
#     #     "description": "overview of the entire field of psychiatry is a staple board review text for psychiatry residents and is popular with a broad range of students in medicine, clinical psychology, social work, nursing, and occupational therapy, as well as practitioners in all these areas.",
#     #     "files": ["index-docs/psych/Synopsis_of_Psychiatry.pdf"],
#     # },
# }
# # initiate agent executor
# args = {
#     "doc_use_qachain": False,
#     "log_tool_selector": False,
#     "use_cache_from_log": False,
# }
# test_agent_executor = AgentExecutorHandler(
#     pipeline=pipeline,
#     embedding=eb,
#     tool_names=test_tool_list,
#     doc_info=test_doc_info,
#     run_tool_selector=True,
#     update_long_term_memory=False,
#     use_long_term_memory=False,
#     **args,
# )
# # start UI
# ui_run = WebUI(test_agent_executor.run)
# ui_run.launch(server_name="0.0.0.0", server_port=7860)

args = {
    "use_cache_from_log": True,
}
chain_config = [
    {
        "name": "task1",
        "type": "simple",
        "input_template": prompts.CHAIN_EXAMPLE_A1,
    },
    {
        "name": "task2",
        "type": "simple",
        "input_template": prompts.CHAIN_EXAMPLE_A2,
    },
    {
        "name": "task3",
        "type": "simple",
        "input_template": prompts.CHAIN_EXAMPLE_A3,
    },
]

custom_chains = ChainSequence(config=chain_config, pipeline=pipeline, **args)
ui_run = WebUI(custom_chains.run)
ui_run.launch(server_name="0.0.0.0", server_port=7860)

print("stop")

# test_prompt = """What did the president say about Ketanji Brown Jackson in his address to the nation?"""
# test_prompt = """Which city will be hosting the summer olympics in 2032?"""
# test_prompt = """Which city will be hosting the summer olympics in 2036?"""
# test_prompt = """What is the current inflation rate in the United States?"""
# test_prompt = """Who leaked the document on Ukraine?"""
# test_prompt = """What is the current progress on nuclear fusion?"""
# test_prompt = """What wars are happening around the world right now?"""
# test_prompt = """Summarize the current events today on the US Economy."""
# test_prompt = """What is the current financial situation of TransLink?"""
# test_prompt = """What are some major benefits of the Millennium Line UBC Extension in Metro Vancouver?"""
# test_prompt = """What are the diagnostic criteria for Intermittent Explosive Disorder?"""
