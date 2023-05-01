# chain example A prompts
CHAIN_EXAMPLE_A1 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Provide a list of names of fictitious companies that make {input}.

### Response:
"""

CHAIN_EXAMPLE_A2 = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please randomly select a single item from the input list. Provide the item only.

### Input:
{task1_output}

### Response:
The selected random item from the given input list is """

CHAIN_EXAMPLE_A3 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is a good slogan for a company that makes {input} and named {task2_output}?

### Response:
"""

# chain example B prompts
CHAIN_EXAMPLE_B1 = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You will be provided with some of your memories. Given the memories provided, please answer this question: {input}?

### Input:
Here are some of my memories from my notes ... {task1_summaries}

### Response:
"""

CHAIN_EXAMPLE_B2 = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
What is a follow up question that you can ask to learn more about from following fact?

### Input:
{task1_output}

### Response:
To learn more about this fact, this is the question I would ask: """

CHAIN_EXAMPLE_B3 = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Given the facts provided, please answer this question: {input}?
In response to this question, you asked the following follow-up question, and obtained information from Google Search below: {task2_output}

### Input:
Information obtained from Google Search ... {task3_summaries}

### Response:
"""
