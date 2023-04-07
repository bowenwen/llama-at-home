# LLaMa at Home

## Model and environment setup

### 1. Install packages for text-generation-webui and langchain

Following instruction from [text-generation-webui](https://github.com/oobabooga/text-generation-webui#manual-installation-using-conda):

For additional package installation, see `environments/langchain.sh`

Not interested in getting the latest development version? Install versions from `environments/langchain.yml`

```bash
conda env create -f ./environments/langchain.yml
```

### 2. Obtain weights

#### Option 1: Download weights

get the 4bit huggingface version 2 (HFv2) from [here](https://rentry.org/llama-tard-v2)

#### Option 2: Convert weights yourself

request the [original facebook weights](https://github.com/facebookresearch/llama). Then convert the weight to HFv2, [detail](https://github.com/oobabooga/text-generation-webui/wiki/LLaMA-model#convert_llama_weights_to_hfpy)

##### Step 1: convert weights to HFv2

```bash
# Get latest PR on weight conversion with llamba from https://github.com/huggingface/transformers
cd packages
mkdir tmp_transformer
git clone -b llama_push https://github.com/zphang/transformers.git
cd transformers
# convert 7B model
python src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ../llama/downloads --model_size 7B --output_dir ../text-generation-webui/models
mv ../text-generation-webui/models/tokenizer/* ../text-generation-webui/models/llama-7b
# convert 13B model
python src/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ../llama/downloads --model_size 13B --output_dir ../text-generation-webui/models
mv ../text-generation-webui/models/tokenizer/* ../text-generation-webui/models/llama-13b
# done!
cd ..
```

##### Step 2: set up for quantized 4bit weights

Quantize to 4bit, read more here: https://github.com/oobabooga/text-generation-webui/wiki/LLaMA-model#4-bit-mode
https://github.com/qwopqwop200/GPTQ-for-LLaMa.

For the most part, it's easier to download the 4-bit weights, but you can also follow Step 3 to convert it yourself.

##### Step 3: convert the llama model to HF and 4 bits

```bash
python convert_llama_weights_to_hf.py --input_dir path_to_model/downloads --model_size 7B --output_dir ../../models_converted
python convert_llama_weights_to_hf.py --input_dir path_to_model/downloads --model_size 13B --output_dir ../../models_converted
python convert_llama_weights_to_hf.py --input_dir path_to_model/downloads --model_size 30B --output_dir ../../models_converted
python convert_llama_weights_to_hf.py --input_dir path_to_model/downloads --model_size 65B --output_dir ../../models_converted
```

The models need to be placed in the `models/` folder, and symbolic linked for text-generation-webui. For example:
```bash
ln -s ~/projects/llama-at-home/models ~/projects/llama-at-home/tools/text-generation-webui/models
ln -s ~/projects/llama-at-home/loras ~/projects/llama-at-home/tools/text-generation-webui/loras
```

### 3. Fine tuning your model

```bash
python finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-alpaca'
```

Examples of other people's fine tuning results
* https://huggingface.co/Draff/llama-alpaca-stuff/tree/main/Alpaca-Loras
* https://huggingface.co/samwit/alpaca13B-lora
* https://huggingface.co/chansung/alpaca-lora-30b

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
python server.py --listen --model llama-13b-4bit-128g --wbits 4 --groupsize 128 -chat

# Running with LoRA
python server.py --listen --model llama-7b  --lora alpaca-lora-7b  --load-in-8bit
python server.py --load-in-8bit --no-stream --model llama-13b --lora alpaca13B-lora --listen

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
