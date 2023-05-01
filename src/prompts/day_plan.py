# prompts to help agent plan its day

CHAIN_DAY_PLAN_FIRST_ACTIVITY = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
The following is a description of yourself:
{identity_statement}

Here are some things you have been thinking about lately:
{recent_memories}

Yesterday, this is what you got up to:
{last_plan}

Your task is to select which activity to do first today. Pick just one activity from the list, providing only the activity name.

### Input:
Here is a list of potential activities to do first today. 

Activity list:
{activity_list}

### Response:
The first activity from the activity list that I will do today is: 
1. """

CHAIN_DAY_PLAN_FIRST_ACTIVITY_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
The following is a description of yourself:
{identity_statement}

Here are some things you have been thinking about lately:
{recent_memories}

Yesterday, this is what you got up to:
{last_plan}

Your task is to choose the topic or area of focus within your first activity.

### Input:
I have chosen to do {first_activity} as my first activity. I need to choose a question, topic, or area of focus within this activity.

### Response:
The question, topic or area focus within {first_activity} that I will focus on is: 
- """

CHAIN_DAY_PLAN_NEXT_ACTIVITY = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
The following is a description of yourself:
{identity_statement}

Here are some things you have been thinking about lately:
{recent_memories}

Yesterday, this is what you got up to:
{last_plan}

Your task is to select which activity to do next today. Pick just one activity from the list, providing only the activity name. If your plan for the day so far already has an activity, you should try to pick a different activity from the activity list.

### Input:
Plan for the day so far:
{plan_under_construction}

Activity list:
{activity_list}

### Response:
The next activity from the activity list that I will do today is: 
"""

CHAIN_DAY_PLAN_NEXT_ACTIVITY_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
The following is a description of yourself:
{identity_statement}

Here are some things you have been thinking about lately:
{recent_memories}

Yesterday, this is what you got up to:
{last_plan}

Here is your plan for the day so far:
{plan_under_construction}

Your task is to choose the topic or area of focus within the next activity.

### Input:
I have chosen to do {next_activity} as my next activity. I need to choose a question, topic, or area of focus within this activity. I need to make sure that I don't repeat anything I have already done in my plan for the day so far. I can feel free to be creative!

### Response:
The question, topic or area focus within {next_activity} that I will focus on is: 
- """

CHAIN_DAY_PLAN_OVERALL_GOAL = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
The following is a description of yourself:
{identity_statement}

Your task is to summarize the overarching theme or goal of today's day plan.

### Input:
Here is your plan for the day:
{day_plan}

### Response:
My goal today is to """
