# LLaMa at Home

## Model and environment setup

Follow this section get your working environment ready to run langchain with llama at home. All of the code, instruction and setup assume you are using gpu (cuda). You may need to review source materials where I referenced to modify the code and instruction to run with cpu.

### 1. Install packages for text-generation-webui and langchain

Following instruction from [text-generation-webui](https://github.com/oobabooga/text-generation-webui#manual-installation-using-conda):

For additional package installation, see `environments/langchain.sh`

Not interested in getting the latest development version? Install versions from `environments/langchain.yml`

```bash
conda env create -f ./environments/langchain.yml
```

### 2. Obtain weights

Before you start, any downloaded or converted models need to be placed in the `models/` folder. Both langchain and text-generation-webui will need to use the models, so you might want to consider using symbolic link to avoid needing to keep two copies, for example:
```bash
ln -s ~/projects/llama-at-home/models ~/projects/llama-at-home/tools/text-generation-webui/models
ln -s ~/projects/llama-at-home/loras ~/projects/llama-at-home/tools/text-generation-webui/loras
ln -s ~/projects/llama-at-home/tools/GPTQ-for-LLaMa ~/projects/llama-at-home/tools/text-generation-webui/repositories/GPTQ-for-LLaMa
```

#### Option 1: Download weights

Get the 4bit huggingface version 2 (HFv2) from [here](https://rentry.org/llama-tard-v2). Downloaded weights only work for a time, until transformer update its code and it will break it eventually. For more future-proof approach, try convert the weights yourself.

#### Option 2: Convert weights yourself

Request the [original facebook weights](https://github.com/facebookresearch/llama). Then convert the weight to HFv2, [detail](https://github.com/oobabooga/text-generation-webui/wiki/LLaMA-model#convert_llama_weights_to_hfpy). Note that since April 2023, [convert_llama_weights_to_hf.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) is now part of the transformer repo!

##### Step 1: convert weights to HuggingFace format

```bash
cd packages/transformers
# convert 7B model
python src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ../../models/facebook --model_size 7B --output_dir ../../models/hfv2
mv ../../models/hfv2 ../../models/llama-7b
# convert 13B model
python src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ../../models/facebook --model_size 13B --output_dir ../../models/hfv2
mv ../../models/hfv2 ../../models/llama-13b
# convert 30B model
python src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ../../models/facebook --model_size 30B --output_dir ../../models/hfv2
mv ../../models/hfv2 ../../models/llama-30b
# if your gpu is good enough to convert 65B model, please do share!
# done, going back to top directory
cd ../..
```

##### Step 2: conversion for quantized 4-bits weights

To conv ert Quantize to 4bit, read more here: https://github.com/oobabooga/text-generation-webui/wiki/LLaMA-model#4-bit-mode
https://github.com/qwopqwop200/GPTQ-for-LLaMa.

For the most part, it is easier to download the 4-bit weights, but you can also follow Step 3 to convert it yourself.

### 3. Fine tuning your model

Read more about fine tuning with alpaca-lora [here](https://github.com/tloen/alpaca-lora#training-finetunepy)

Train lora for llama-7b:

```bash
cd tools/alpaca-lora
python finetune.py \
    --base_model="../../models/llama-7b" \
    --data_path="alpaca_data_gpt4.json" \
    --num_epochs=10 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir="../../loras/alpaca-lora-7b" \
    --lora_target_modules="[q_proj,k_proj,v_proj,o_proj]" \
    --lora_r=16 \
    --micro_batch_size=8
cd ../..
```

Train lora for llama-13b:

```bash
cd tools/alpaca-lora
python finetune.py \
    --base_model='../../models/llama-13b' \
    --num_epochs=10 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='../../loras/alpaca-lora-13b' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r=16 \
    --batch_size=128  \
    --micro_batch_size=4
cd ../..
```

Examples of other fine tuning results, as with the converted weights, downloaded loras may not work with the latest versions of various packages:
* https://github.com/tloen/alpaca-lora#resources
* https://huggingface.co/tloen/alpaca-lora-7b
* https://huggingface.co/chansung/alpaca-lora-13b
* https://huggingface.co/chansung/alpaca-lora-30b
* https://huggingface.co/chansung/alpaca-lora-65b

### 4. Configuring model parameters for alpaca lora

Commonly, HuggingFace models will come with a `config.json` which you can use to add your own configurations, some common parameters are shown before. See [HuggingFace Documentation] for more.

```json
{
  "do_sample": 1,
  "temperature": 0.7,
  "top_p": 0.1,
  "typical_p": 1,
  "repetition_penalty": 1.18,
  "encoder_repetition_penalty": 1,
  "top_k": 40,
  "num_beams": 1,
  "penalty_alpha": 0,
  "min_length": 0,
  "length_penalty": 1,
  "no_repeat_ngram_size": 0,
  "early_stopping": 0,
}
```

* https://huggingface.co/blog/how-to-generate
* https://docs.cohere.ai/docs/controlling-generation-with-top-k-top-p

### 5. Finetuning 4-bit models

The process for finetuning 4-bit models is slight more complicated and require patching of some existing models. See [johnsmith0031/alpaca_lora_4bit](https://github.com/johnsmith0031/alpaca_lora_4bit) for more information. Note that with this set up, you may not be able to use 8 bit lora properly within reinstalling official version of peft and GPTQ-for-LLaMa.

### Installation

```
cd tools/alpaca_lora_4bit
pip install -r requirements.txt
```

### Other models

#### Trying out Cerebras-GPT

* Source: https://huggingface.co/cerebras

bash
```
cd ~/apps/vscodeserver/appdata/coder/projects/Cerebras-GPT
git lfs install
git clone https://huggingface.co/cerebras/Cerebras-GPT-111M
git clone https://huggingface.co/cerebras/Cerebras-GPT-256M
git clone https://huggingface.co/cerebras/Cerebras-GPT-590M
git clone https://huggingface.co/cerebras/Cerebras-GPT-1.3B
git clone https://huggingface.co/cerebras/Cerebras-GPT-2.7B
git clone https://huggingface.co/cerebras/Cerebras-GPT-6.7B
git clone https://huggingface.co/cerebras/Cerebras-GPT-13B
```


## Using text-generation-webui

```bash
conda activate langchain

# use 8 bits
python server.py --listen --load-in-8bit --no-stream --model llama-7b
python server.py --listen --load-in-8bit --no-stream --model llama-13b

# Or, use 4 bits
python server.py --listen --model llama-7b-4bit-128g --wbits 4 --groupsize 128 --no-stream --chat
python server.py --listen --model llama-13b-4bit-128g --wbits 4 --groupsize 128 --chat
python server.py --listen --model llama-30b-4bit-128g --wbits 4 --groupsize 128 --chat

# Running with LoRA
python server.py --listen --model llama-7b --lora alpaca-lora-7b --load-in-8bit
python server.py --listen --model llama-13b --lora alpaca-lora-13b --load-in-8bit

# Starting API (same with regular but no chat), api at `http://{server}:7860/api/textgen`
python server.py --listen --listen-port 7860 --load-in-8bit --no-stream --model llama-7b
# --extensions api
```

You may need to update the model name if the downloaded version is slightly different.

Browse to: `http://localhost:7860/?__theme=dark`
[help?](https://github.com/oobabooga/text-generation-webui#starting-the-web-ui)

## Using LangChain

Set up documents

```bash
mkdir -p ./docs/arxiv/
cp /homenas/media/coder/projects/LLM-documents/arxiv/* ./docs/arxiv/
```

Some reading materials on Langchain
* https://huggingface.co/blog/hf-bitsandbytes-integration
* https://huggingface.co/docs/transformers/v4.13.0/en/performance


# Notes

```
conda env export --no-builds > ./environments/langchain.yml
```

## Resources
* make it work with UI: https://rentry.org/llama-tard-v2
* Fine tuning: https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama
* CPP implementation: https://github.com/ggerganov/llama.cpp
* FlexGen: https://github.com/FMInference/FlexGen
* https://github.com/intel/intel-extension-for-pytorch
