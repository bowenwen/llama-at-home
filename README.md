# LLaMa at Home

## Getting Started

### 1. Set up the basic environment for llama, skip if using HFv2 weights directly

```bash
conda remove --name llama --all
conda create -n llama "python>=3.10"
conda activate llama
git clone https://github.com/facebookresearch/llama.git
cd llama
pip install -r requirements.txt
pip install -e .
cd ..
```

### 2. Install additional packages for textgen UI

```bash
conda remove --name textgen --all
conda create -n textgen python=3.10.9
conda activate textgen
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
conda install torchvision=0.14.1 torchaudio=0.13.1 pytorch-cuda=11.7 git -c pytorch -c nvidia
# get text gen webui
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r requirements.txt
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers
```

#### dirty fix required for linux, use if the above install did not work

[source](https://github.com/oobabooga/text-generation-webui/issues/400#issuecomment-1474876859)

```
cd ~/conda/envs/textgen/lib/python3.10/site-packages/bitsandbytes/
cp libbitsandbytes_cuda120.so libbitsandbytes_cpu.so
conda install cudatoolkit
cd ~/projects/LLaMA/text-generation-webui
```

### 3. Obtain weights

Option 1: get the 4bit huggingface version 2 (HFv2) from [here](https://rentry.org/llama-tard-v2)

Option 2: request the [original facebook weights](https://github.com/facebookresearch/llama). Then convert the weight to HFv2, [detail](https://github.com/oobabooga/text-generation-webui/wiki/LLaMA-model#convert_llama_weights_to_hfpy)

- Step 1: convert weights to HFv2

```bash
# Get latest PR on weight conversion with llamba from https://github.com/huggingface/transformers
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

- Step 2: set up for quantized 4bit weights

Quantize to 4bit, read more: https://github.com/oobabooga/text-generation-webui/wiki/LLaMA-model#4-bit-mode
https://github.com/qwopqwop200/GPTQ-for-LLaMa

```bash
mkdir repositories
cd repositories
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa
cd GPTQ-for-LLaMa
git checkout 468c47c01b4fe370616747b6d69a2d3f48bab5e4
python setup_cuda.py install
```

- Step 3: convert the llama model to HF and 4 bits

```bash
python convert_llama_weights_to_hf.py --input_dir /homenas/media/coder/projects/LLaMA_models/downloads --model_size 7B --output_dir ../../models_converted
python convert_llama_weights_to_hf.py --input_dir /homenas/media/coder/projects/LLaMA_models/downloads --model_size 13B --output_dir ../../models_converted
python convert_llama_weights_to_hf.py --input_dir /homenas/media/coder/projects/LLaMA_models/downloads --model_size 30B --output_dir ../../models_converted
python convert_llama_weights_to_hf.py --input_dir /homenas/media/coder/projects/LLaMA_models/downloads --model_size 65B --output_dir ../../models_converted
```


### 4. Start the UI

```bash
conda activate textgen

# use 8 bits
python server.py --load-in-8bit --no-stream --model llama-13b
# python server.py --listen --model llama-7b  --lora alpaca-lora-7b  --load-in-8bit

# Or, use 4 bits
python server.py --gptq-bits 4 --no-stream --model llama-7b --chat
python server.py --gptq-bits 4 --no-stream --model llama-13b --chat
python server.py --gptq-bits 4 --no-stream --model llama-30b --chat

python server.py --load-in-8bit --no-stream --model llama-13b --lora alpaca13B-lora --listen --chat

# Running with LoRA
python server.py --load-in-8bit --no-stream --model llama-13b --lora alpaca13B-lora --listen
python server.py --gptq-bits 4 --no-stream --model llama-13b --lora alpaca13B-lora
python server.py --gptq-bits 4 --no-stream --model llama-30b --lora alpaca-lora-30b
# python server.py --gptq-bits 4 --no-stream --model llama-30b --lora alpaca-lora-30b

# Starting API (same with regular but no chat), api at `http://{server}:7860/api/textgen`
python server.py --listen --listen-port 7860 --load-in-8bit --no-stream --model llama-7b
# --extensions api
```

You may need to update the model name if the downloaded version is slightly different.

Browse to: `http://localhost:7860/?__theme=dark`
[help?](https://github.com/oobabooga/text-generation-webui#starting-the-web-ui)


### 5. Fine tuning

```
conda activate textgen
cd ~/projects/LLaMA
git clone https://github.com/tloen/alpaca-lora.git
cd alpaca-lora
pip install -r requirements.txt
```

#### Examples of other people's fine tuning results
- https://huggingface.co/Draff/llama-alpaca-stuff/tree/main/Alpaca-Loras
- https://huggingface.co/samwit/alpaca13B-lora
- https://huggingface.co/chansung/alpaca-lora-30b


### 6. Build your own server for LLaMa Chat!

```bash
git clone https://github.com/bowenwen/text-generation-docker.git
docker build -t text-generation-docker .

docker run -it --rm --name llamba-server -d \
  text-generation-docker:latest


docker run -it --rm --name code-server -p 127.0.0.1:8080:8080 \
  -v "$HOME/.config:/home/coder/.config" \
  -v "$PWD:/home/coder/project" \
  -u "$(id -u):$(id -g)" \
  -e "DOCKER_USER=$USER" \
  code-server-miniconda3:latest
```


## Resources
* make it work with UI: https://rentry.org/llama-tard-v2
* Fine tuning: https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama
* CPP implementation: https://github.com/ggerganov/llama.cpp
* FlexGen: https://github.com/FMInference/FlexGen
* https://github.com/intel/intel-extension-for-pytorch

Source location of all models downloaded: `/homenas/media/coder/projects/LLaMA_models`
