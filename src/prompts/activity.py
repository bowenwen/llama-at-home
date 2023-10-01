PLAN_SUBTASK_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Today, your overall goal is to {overall_goal}. To that end, you are currently engaging in the {activity_type} activity.

Here are some recent memories from your notes:
{recent_memories}

Your task is to take the prompt/question in the input and come up with 3 distinct follow-up prompts/questions which elaborate or cover different aspects of the prompt. The response should be a numbered list. 

### Input:
{activity_input}

### Response:
1. """
