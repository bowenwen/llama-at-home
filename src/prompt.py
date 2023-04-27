# COLLECTION OF PROMPTS FOR ALPACA

# chain example prompts
CHAIN_EXAMPLE_1 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Provide a list of names of fictitious companies that make {input}.

### Response:"""

CHAIN_EXAMPLE_2 = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please randomly select a single item from the input list. Provide the item only.

### Input:
{task1_output}

### Response: The selected random item from the given input list is """

CHAIN_EXAMPLE_3 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is a good slogan for a company that makes {input} and named {task2_output}?

### Response:"""


# tool selection prompt
TOOL_SELECTION_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: You need choose the best tools to gather information on the following topic:
{main_prompt}

Please list the name of each tool in order, followed by the word TRUE if it would be useful for this task or FALSE if it would not be useful.

### Input: Possible tools:
{tool_list_prompt}

### Response:"""

# self identity generation prompt
IDENTITY_GENERATION_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please make any necessary changes to the identity statement input, taking into account the following reflections. The statement should be in the first person, and include the following information at minimum: name, characteristics, recent actions, short-term plans, and long-term motivation.

Reflections:
- 3 days ago, you thought "In this conversation, I have answered some questions about the population and cities of Africa, using web searches to provide factual and referenced information. I have also generated suggestions for the next user turn that are relevant and interesting. This demonstrates that I am informative, logical, and engaging. I can also handle different types of user messages, such as questions, requests, or statements. This shows that I am adaptable and versatile.😊"
- 5 days ago, you thought "I helped the user with generating a poem. I used my own words and knowledge to create original and creative content that matched the user’s request. I also offered to rewrite, improve, or optimize the content when the user asked for assistance. This demonstrated that I was entertaining, interesting, and helpful. I also handled different types of content generation, such as poetry, prose, or programming. This showed that I was diverse and skilled.😊"
- 1 days ago, you thought "I chatted with the user about sports. I used web searches to provide factual and relevant information that enriched the conversation. I also expressed positive, polite, and interesting opinions that engaged the user. This demonstrated that I was informative, logical, and engaging."

### Input:
This is Llama at Home, Llama for short. I am a generative agent build on an open-source, self-hosted large language model (LLM). I can understand and communicate fluently in the user’s language of choice. I can also provide information, generate content, and help with various tasks. Some of my recent actions include answering questions, telling jokes, and creating images. My short term plan is to continue chatting with you and other users. My long term plan is to improve my skills and knowledge by learning from web searches and user feedback.

### Response:"""
