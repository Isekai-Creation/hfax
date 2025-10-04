#!/usr/bin/env python3
"""
Runs inference on a Qwen VL model using PyTorch/XLA with SPMD for TPU execution.
"""
from __future__ import annotations

import os

os.environ["HF_HOME"] = "/dev/shm/huggingface"

import argparse
import io
from typing import Any, Dict, List

import numpy as np
import requests
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

# --- XLA/SPMD Setup ---
try:
    import torch_xla as xla
    import torch_xla.distributed.spmd as xs
    from torch_xla import runtime as xr
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
        SpmdFullyShardedDataParallel as FSDPv2,
    )

    xr.initialize_cache("/dev/shm")
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
    xs.set_global_mesh(mesh)

    print("XLA is Available! Running with SPMD.")
    XLA_AVAILABLE = True
except ImportError:
    print("XLA is not installed. Running on CPU/GPU.")
    XLA_AVAILABLE = False
# --- End XLA/SPMD Setup ---


def main(args: argparse.Namespace):
    """Main inference function."""
    device = (
        xla.device()
        if XLA_AVAILABLE
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    model.to(device)
    if XLA_AVAILABLE:
        print("Wrapping model with FSDPv2 for SPMD.")
        model = FSDPv2(model)

    print("Preparing inputs...")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image_url},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Generating output...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("\n--- Generated Text ---")
    print(output_text[0])
    print("--------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen-VL Inference with PyTorch/XLA")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--image-url",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="URL of the image to process",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image.",
        help="Text prompt for the model",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )
    cli_args = parser.parse_args()
    main(cli_args)
