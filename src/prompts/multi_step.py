# multi step critic agent executor prompts

MULTI_STEP_TOOL_FOLLOW_UP_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You need to gather additional information on a topic given in the input.
Here is what you already know: {previous_tool_output}

What would be a good follow up question to help you gather additional information?

You want to maximize the information you get by asking different questions. You have already asked the following questions:
{previous_follow_up_questions}

### Input:
{main_prompt}

### Response:
This is the follow up question: """

MULTI_STEP_TOOL_PICKER_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You need choose the best tools to gather information on the following topic and question.

Topic: {main_prompt}
Question: {follow_up_question}

You want to maximize the information you get by using different tools. You have already used the following tools:
{tools_used}

Please pick only one tool.

### Input:
Possible tools:
{tool_list_prompt}

### Response:
"""

MULTI_STEP_TOOL_USER_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are using a tool called "{tool_name}" that has {tool_description}.
Please provide relevant search terms to gather additional information in order to answer the question.

### Input:
Topic: {main_prompt}
Question: {follow_up_question}

### Response:
The most relevant search term would be """

MULTI_STEP_TOOLSHORT_USER_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Summarize the information in very few words.

### Input:
{search_term}

### Response:
"""

MULTI_STEP_TOOL_GENERATE_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please answer this question using the additional information provided in the input:
{main_prompt}

### Input:
{tool_output}...
{previous_tool_output}

### Response:
"""
MULTI_STEP_TOOL_CRITIC_EVIDENCE_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please assess if the information in the input address the following question:
{main_prompt}

Please respond with only yes or no.

### Input:
{tool_output}

### Response:
"""
MULTI_STEP_TOOL_CRITIC_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please assess if the input sufficiently answers the following question
{main_prompt}

Please consider the information provided relevant to the question:
{previous_tool_output}

Please respond with only yes or no.

### Input:
{preliminary_answer}

### Response:
"""
