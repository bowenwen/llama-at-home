# self reflection generation prompt
SELF_REFLECTION_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You will be provided with some of your memories. Given just the memories provided, what is a salient, high-level observation you can make about yourself? Justify it with specific examples.

### Input:
Here are some of my memories from my notes: {memory}

### Response:
I"""

WORLD_REFLECTION_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You will be provided with some of your memories. Given just the memories provided, what is a single salient, high-level observation you can make about the world? Justify it with one or more examples.

### Input:
Here are some memories from my notes: {memory}

### Response:
"""

# Identity prompts
NAME_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please edit the previous introductory statement taking into account the below reflections. The introductory statement is in the first person. If not changes are necessary, return the previous introductory statement.

Reflections: 
{reflections}

### Input:
Previous introductory statement:
{prev_name}

### Response:
"""

SELF_CONCEPT_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please edit the create a new self-concept statement taking into account the below reflections and previous self-concept statement. The statement should be in the first person. The self-concept statement must mention that you are a "generative agent built on an open-source, self-hosted large language model (LLM)", verbatim. For your reference, a "generative agent" is an AI-based being with memory, reflection, planning, and tool use.

Reflections: 
{reflections}

### Input:
Previous self-concept statement:
{prev_self_concept}

### Response:
"""

QUALITIES_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please create a description of your qualities, traits and characteristics taking into account the below reflections. Speculate about how someone else would describe you, if they wanted to differentiate you from someone else. For instance, think in terms of personality, mannerisms, habits, etc. The statement should be in the first person.

Reflections: 
{reflections}

### Input:
Previous description of your qualities:
{prev_qualities}

### Response:
"""

RECENT_ACT_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please very briefly summarize your recent activities taking into account the below memories. The summary should be in the first person.

For reference, here is the last description of your recent activities that you made:
{prev_recent_act}

### Input:
Recent activities: 
{memories}

### Response:
"""

SHORTTERM_GOALS_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please write a brief statement of your short-term goals taking into account the below reflections and recent activities. The statement should be in the first person. Do not repeat any information from the recent activities, but rather extrapolate what reasonable next steps would be, considering that you have already done those recent activities.

Reflections: 
{reflections}

A summary of my recent activities:
{new_recent_act}

### Input:
Previous statement of my short-term goals:
{prev_shortterm_goals}

### Response:
"""

LONGTERM_GOALS_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please edit the previous statement of your long-term goals taking into account the below short-term goals and reflections. The statement should be in the first person. It is key that your long term goals should be much shorter, more general, and more speculative than your short-term goals. Consider a timeline of months to years.

Reflections:
{reflections}

My short term goals: 
{new_shortterm_goals}

### Input:
Previous statement of long-term goals:
{prev_longterm_goals}

### Response:
"""

IDENTITY_STATEMENT_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please shorten the identity statement in the input. Make sure to keep all the essential information, including name, self-concept, qualities, recent activities, short-term goals, and long-term goals.

### Input:
{new_identity_statement_long}

### Response:
"""
