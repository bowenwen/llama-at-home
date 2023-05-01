# multi step critic agent executor prompts

MULTI_STEP_TOOL_FOLLOW_UP_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You need to gather additional information on a topic given in the input.
Here is what you already know: {previous_tool_output}

What would be a good follow up question to help you gather additional information?

### Input:
{main_prompt}

### Response:
This is the follow up question: """

MULTI_STEP_TOOL_PICKER_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You need choose the best tools to gather information on the following topic:
{follow_up_question}

You also want to maximize the information you get by using different tools. You have already used the following tools:
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
{follow_up_question}

### Response:
The most relevant search term would be """

MULTI_STEP_TOOL_GENERATE_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please answer this question using additional information provided in the input:
{main_prompt}

### Input:
Recently obtained information - {tool_output}
Information from my previous notes - {previous_tool_output}

### Response:
"""
MULTI_STEP_TOOL_CRITIC_EVIDENCE_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Does the input address the following question?
{main_prompt}

Please repond in yes or no.

### Input:
{tool_output}

### Response:
"""
MULTI_STEP_TOOL_CRITIC_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Does the input sufficiently answers the following question?
{main_prompt}

Please consider the information provided relevant to the question:
{previous_tool_output}

Please repond in yes or no.

### Input:
{preliminary_answer}

### Response:
"""
