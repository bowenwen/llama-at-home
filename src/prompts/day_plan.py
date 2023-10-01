# prompts to help agent plan its day

FIRST_PLAN_GENERATOR_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
The following is a description of yourself:
{identity_statement}

Here are some things you have been thinking about lately:
{recent_memories}

Your task is to take the day plan in the Input and substitute every reference to penguins, flightless birds, etc. with any other topic at all, of your choosing. It is important to maintain the format of the plan (numbers, brackets, etc.). Do not change the words Research, Write, News, or Reflect. Only change the text within the square brackets.

### Input:
1. Research[Penguins and their characteristics]
2. Write[Poem about the antarctic]
3. Research[Flightless birds, general]
4. News[scientific news, with a focus on biology]
5. Reflect[penguins and what they mean to me]

### Response:
1. Research["""

PLAN_REPETITION_REMOVAL_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
The following is a description of yourself:
{identity_statement}

Your task is to take the day plan in the Input and slightly change the text within the brackets which is repeated between lines. It is important to maintain the format of the plan (numbers, brackets, etc.). Do not change the words Research, Write, News, or Reflect. Only change the text within the square brackets. 

For example, if it says 1. Research[cats], 2. Research[cats]... you could change this to 1. Research[cats], 2. Research[felines].

### Input:
{plan_under_construction}

### Response:
1. """


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
Activity list:
{activity_list}

### Response:
The name of the from the activity list that I will do first today is: 
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
The question, topic or area of focus within {first_activity} that I will focus on is: 
- """

CHAIN_DAY_PLAN_NEXT_ACTIVITY = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
The following is a description of yourself:
{identity_statement}

Here are some things you have been thinking about lately:
{recent_memories}

Yesterday, this is what you got up to:
{last_plan}

Your task is to select which activity to do next today. Pick just one activity from the list, providing only the activity name.

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

Activity list:
{activity_list}

Your task is to choose a simple new question, topic or area of focus. It should be different from the previous topic, and related to the activity. 

### Input:
Next activity: {next_activity} 

Previous question, topic or area of focus:
- "{prev_activity_input}"

### Response:
My new question, topic or area of focus for the "{next_activity}" activity will be:
- """

CHAIN_DAY_PLAN_OVERALL_GOAL = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Your task is to express a single major overarching theme or goal of today's day plan, by finding a single common denominator between the different parts of the plan.

### Input:
Here is my plan for the day:
{day_plan}

### Response:
My goal today is to """
