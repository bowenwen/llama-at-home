import inspect
import re
import sys
from pathlib import Path

import accelerate
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path("tools/GPTQ-for-LLaMa")))
import llama_inference_offload
from modelutils import find_layers
from quant import make_quant

import argparse

# source: https://github.com/oobabooga/text-generation-webui
# modified to work with llama-at-home


def _load_quant(
    model,
    checkpoint,
    wbits,
    groupsize=-1,
    faster_kernel=False,
    exclude_layers=["lm_head"],
    kernel_switch_threshold=128,
):
    def noop(*args, **kwargs):
        pass

    config = AutoConfig.from_pretrained(model)
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in exclude_layers:
        if name in layers:
            del layers[name]

    gptq_args = inspect.getfullargspec(make_quant).args

    make_quant_kwargs = {
        "module": model,
        "names": layers,
        "bits": wbits,
    }
    if "groupsize" in gptq_args:
        make_quant_kwargs["groupsize"] = groupsize
    if "faster" in gptq_args:
        make_quant_kwargs["faster"] = faster_kernel
    if "kernel_switch_threshold" in gptq_args:
        make_quant_kwargs["kernel_switch_threshold"] = kernel_switch_threshold

    make_quant(**make_quant_kwargs)

    del layers

    print("Loading model ...")
    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load

        model.load_state_dict(safe_load(checkpoint), strict=False)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)
    model.seqlen = 2048
    print("Done.")

    return model


def load_quantized(model_name, **kwargs):
    # set arguments, originally shared.args
    # GPTQ
    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=54)
    )
    parser.add_argument(
        "--wbits",
        type=int,
        # default=0,
        default=kwargs["wbits"],
        help="GPTQ: Load a pre-quantized model with specified precision in bits. 2, 3, 4 and 8 are supported.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="GPTQ: Model type of pre-quantized model. Currently LLaMA, OPT, and GPT-J are supported.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        # default=-1,
        default=kwargs["groupsize"],
        help="GPTQ: Group size.",
    )
    parser.add_argument(
        "--pre_layer",
        type=int,
        # default=0,
        default=kwargs["pre_layer"],
        help="GPTQ: The number of layers to allocate to the GPU. Setting this parameter enables CPU offloading for 4-bit models.",
    )
    parser.add_argument(
        "--gptq-bits", type=int, default=0, help="DEPRECATED: use --wbits instead."
    )
    parser.add_argument(
        "--gptq-model-type", type=str, help="DEPRECATED: use --model_type instead."
    )
    parser.add_argument(
        "--gptq-pre-layer",
        type=int,
        default=0,
        help="DEPRECATED: use --pre_layer instead.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="",
    )
    parser.add_argument(
        "--cpu",
        type=bool,
        default=False,
        # action="store_true",
        help="Use the CPU to generate text.",
    )
    parser.add_argument(
        "--gpu-memory",
        type=str,
        nargs="+",
        help="Maxmimum GPU memory in GiB to be allocated per GPU. Example: --gpu-memory 10 for a single GPU, --gpu-memory 10 5 for two GPUs. You can also set values in MiB like --gpu-memory 3500MiB.",
    )
    gptq_args = parser.parse_args()

    if not gptq_args.model_type:
        # Try to determine model type from model name
        name = model_name.lower()
        if any((k in name for k in ["llama", "alpaca", "vicuna"])):
            model_type = "llama"
        elif any((k in name for k in ["opt-", "galactica"])):
            model_type = "opt"
        elif any((k in name for k in ["gpt-j", "pygmalion-6b"])):
            model_type = "gptj"
        else:
            print(
                "Can't determine model type from model name. Please specify it manually using --model_type "
                "argument"
            )
            exit()
    else:
        model_type = gptq_args.model_type.lower()

    if gptq_args.pre_layer and model_type == "llama":
        load_quant = llama_inference_offload.load_quant
    elif model_type in ("llama", "opt", "gptj"):
        if gptq_args.pre_layer:
            print(
                "Warning: ignoring --pre_layer because it only works for llama model type."
            )
        load_quant = _load_quant
    else:
        print(
            "Unknown pre-quantized model type specified. Only 'llama', 'opt' and 'gptj' are supported"
        )
        exit()

    # Now we are going to try to locate the quantized model file.
    path_to_model = Path(f"{gptq_args.model_dir}/{model_name}")
    found_pts = list(path_to_model.glob("*.pt"))
    found_safetensors = list(path_to_model.glob("*.safetensors"))
    pt_path = None

    if len(found_pts) == 1:
        pt_path = found_pts[0]
    elif len(found_safetensors) == 1:
        pt_path = found_safetensors[0]
    else:
        if path_to_model.name.lower().startswith("llama-7b"):
            pt_model = f"llama-7b-{gptq_args.wbits}bit"
        elif path_to_model.name.lower().startswith("llama-13b"):
            pt_model = f"llama-13b-{gptq_args.wbits}bit"
        elif path_to_model.name.lower().startswith("llama-30b"):
            pt_model = f"llama-30b-{gptq_args.wbits}bit"
        elif path_to_model.name.lower().startswith("llama-65b"):
            pt_model = f"llama-65b-{gptq_args.wbits}bit"
        else:
            pt_model = f"{model_name}-{gptq_args.wbits}bit"

        # Try to find the .safetensors or .pt both in the model dir and in the subfolder
        for path in [
            Path(p + ext)
            for ext in [".safetensors", ".pt"]
            for p in [
                f"{gptq_args.model_dir}/{pt_model}",
                f"{path_to_model}/{pt_model}",
            ]
        ]:
            if path.exists():
                print(f"Found {path}")
                pt_path = path
                break

    if not pt_path:
        print(
            "Could not find the quantized model in .pt or .safetensors format, exiting..."
        )
        exit()

    # qwopqwop200's offload
    if model_type == "llama" and gptq_args.pre_layer:
        model = load_quant(
            str(path_to_model),
            str(pt_path),
            gptq_args.wbits,
            gptq_args.groupsize,
            gptq_args.pre_layer,
        )
    else:
        threshold = False if model_type == "gptj" else 128
        model = load_quant(
            str(path_to_model),
            str(pt_path),
            gptq_args.wbits,
            gptq_args.groupsize,
            kernel_switch_threshold=threshold,
        )

        # accelerate offload (doesn't work properly)
        if gptq_args.gpu_memory:
            memory_map = list(map(lambda x: x.strip(), gptq_args.gpu_memory))
            max_cpu_memory = (
                gptq_args.cpu_memory.strip()
                if gptq_args.cpu_memory is not None
                else "99GiB"
            )
            max_memory = {}
            for i in range(len(memory_map)):
                max_memory[i] = (
                    f"{memory_map[i]}GiB"
                    if not re.match(".*ib$", memory_map[i].lower())
                    else memory_map[i]
                )
            max_memory["cpu"] = max_cpu_memory

            device_map = accelerate.infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=["LlamaDecoderLayer"],
            )
            print("Using the following device map for the 4-bit model:", device_map)
            # https://huggingface.co/docs/accelerate/package_reference/big_modeling#accelerate.dispatch_model
            model = accelerate.dispatch_model(
                model, device_map=device_map, offload_buffers=True
            )

        # No offload
        elif not gptq_args.cpu:
            model = model.to(torch.device("cuda:0"))

    return model
