import sys
import os

import gradio as gr

sys.path.append("./")
from src.my_langchain_models import MyLangchainLlamaModelHandler
from src.my_langchain_agent_executor import MyLangchainAgentExecutorHandler


# run the current implementation of llama-at-home

# set environment variable to save chat history
os.environ["MYLANGCHAIN_SAVE_CHAT_HISTORY"] = "1"

# select model and lora
model_name = "llama-13b"
lora_name = "alpaca-gpt4-lora-13b-3ep"
# model_name = "llama-7b"
# lora_name = "alpaca-lora-7b"

testAgent = MyLangchainLlamaModelHandler()
eb = testAgent.load_hf_embedding()
pipeline, model, tokenizer = testAgent.load_llama_llm(
    model_name=model_name, lora_name=lora_name, max_new_tokens=200
)
# eb = testAgent.load_llama_cpp_embedding(model_name=model_name)
# pipeline = testAgent.load_llama_cpp_llm(
#     model_name=model_name, lora_name=lora_name, max_new_tokens=200, quantized=False
# )

# define tool list (excluding any documents)
test_tool_list = ["wiki", "searx"]

# define test documents
test_doc_info = {
    # "examples": {
    #     "tool_name": "State of Union QA system",
    #     "description": "specific facts from the 2023 state of the union on Joe Biden's plan to rebuild the economy and unite the nation.",
    #     "files": ["index-docs/examples/state_of_the_union.txt"],
    # },
    # "arxiv": {
    #     "tool_name": "Arxiv Papers",
    #     "description": "scientific papers from arxiv on math, science, and computer science.",
    #     "files": [
    #         "index-docs/arxiv/2302.13971.pdf",
    #         "index-docs/arxiv/2304.03442.pdf",
    #     ],
    # },
    "translink": {
        "tool_name": "Translink Reports",
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
    "psych_dsm": {
        "tool_name": "Psychiatry DSM",
        "description": "Diagnostic and Statistical Manual of Mental Disorders helps clinicians and researchers define and classify mental disorders, which can improve diagnoses, treatment, and research.",
        "files": ["index-docs/psych/DSM-5-TR.pdf"],
    },
    # "psych_sop": {
    #     "tool_name": "Synopsis of Psychiatry",
    #     "description": "overview of the entire field of psychiatry is a staple board review text for psychiatry residents and is popular with a broad range of students in medicine, clinical psychology, social work, nursing, and occupational therapy, as well as practitioners in all these areas.",
    #     "files": ["index-docs/psych/Synopsis_of_Psychiatry.pdf"],
    # },
}

# initiate agent executor
args = {"doc_use_qachain": False}
test_agent_executor = MyLangchainAgentExecutorHandler(
    hf=pipeline, tool_names=test_tool_list, doc_info=test_doc_info, **args
)


def run_with_logs(prompt):
    # clear log
    with open("logs/output_now.log", "w") as f:
        print("", file=f)
    # run get to answer
    answer = test_agent_executor.run(prompt)
    # obtain log
    with open("logs/output_now.log", "r") as f:
        # optioanlly, read external log output from langchain
        # require modifying packages/langchain/langchain/input.py
        langchain_log = f.read()
    if os.getenv("MYLANGCHAIN_SAVE_CHAT_HISTORY") == "1":
        with open("logs/output_recent.log", "a") as f:
            print(f"{answer}\n", file=f)
    return [langchain_log, answer]


gr.Interface(
    fn=run_with_logs,
    inputs=["text"],
    outputs=["textbox", "textbox"],
).queue().launch(server_name="0.0.0.0", server_port=7860)

print("stop")

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
