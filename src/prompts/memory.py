# COLLECTION OF PROMPTS FOR ALPACA

# importance rating prompt
IMPORTANCE_RATING_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is
extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory.

### Input:
{memory}

### Response:
Here is the importance rating out of 10: """

# self identity generation prompt
IDENTITY_GENERATION_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Please edit the previous identity statement, taking into account the input reflections. The statement should be in the first person, and include the following information at minimum: name, characteristics, recent actions, short-term plans, and long-term motivation.

Previous identity statement:
{prev_identity}

### Input:
Reflections: {reflections}

### Response:
"""

# self reflection generation prompt
REFLECTION_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You will be provided with some of your memories. Given just the memories provided, what is a salient, high-level observation you can make about yourself as the person having these thoughts. Justify it with specific examples.

### Input:
Here are some memories from my notes: {memory}

### Response:
I"""
