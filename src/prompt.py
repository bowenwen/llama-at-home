# custom prompt for mylangchain
TOOL_SELECTION_PROMPT = """Below is an instruction that describes a task. 

### Instruction: You need choose the best tools to gather information on the following topic:
{main_prompt}

Please list the name of each tool in order, followed by the word TRUE if it would be useful for this task or FALSE if it would not be useful.

### Input: Possible tools:
{tool_list_prompt}

### Response:"""

