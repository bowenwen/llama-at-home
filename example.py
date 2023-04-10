from pathlib import Path
from peft import PeftModelForCausalLM
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

# the pipeline argument required to work with LORA
pipeline_args = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.1,
    "typical_p": 1.0,
    "repetition_penalty": 1.18,
    "encoder_repetition_penalty": 1,
    "top_k": 40,
    "min_length": 0,
    "no_repeat_ngram_size": 0,
    "num_beams": 1,
    "penalty_alpha": 0,
    "length_penalty": 1,
    "early_stopping": False,
    "eos_token_id": [2],
    "stopping_criteria": [],
}
tokenizer = LlamaTokenizer.from_pretrained(
    Path("models/llama-7b"), clean_up_tokenization_spaces=True
)
model = LlamaForCausalLM.from_pretrained(
    Path("models/llama-7b"),
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
)
model = PeftModelForCausalLM.from_pretrained(
    model,
    Path("loras/alpaca-lora-7b"),
    dtype=model.dtype,
    device_map={"base_model.model.": 0},
)

pipe = pipeline(
    model=model,
    task="text-generation",
    framework="pt",
    tokenizer=tokenizer,
    max_new_tokens=50,
    **pipeline_args,
)

text_gen_example = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Write a poem about the transformers Python library. 
Mention the word "large language models" in that poem.
### Response:"""
print(pipe(text_gen_example)[0]["generated_text"])


####################
# EXAMPLE RESPONSE
####################
# Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# Write a poem about the transformers Python library.
# Mention the word "large language models" in that poem.
# ### Response:
# Transformers are large and mighty,
# They can build up your model's might;
# With their power they can make it right,
# And help you to create a light!
