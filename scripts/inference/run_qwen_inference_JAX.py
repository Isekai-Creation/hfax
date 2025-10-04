#!/usr/bin/env python3
"""
Runs inference on a Gemma model using the hfax library.
"""
from __future__ import annotations

import argparse
import io

import hfax
import jax
import numpy as np
import requests
from PIL import Image


def main(args: argparse.Namespace):
    """Main inference function."""
    print("Loading model, tokenizer, and parameters...")

    # Using Gemma model as the Qwen model is not found
    model = hfax.nn.Gemma3_4B()
    ckpt_path = hfax.ckpts.CheckpointPath.GEMMA3_4B_IT
    tokenizer = hfax.text.Gemma3Tokenizer()

    print(f"Loading parameters from: {ckpt_path}")
    params = hfax.ckpts.load_params(path=ckpt_path)

    print("Instantiating ChatSampler...")
    sampler = hfax.text.ChatSampler(
        model=model,
        params=params,
        tokenizer=tokenizer,
    )

    print("Preparing inputs...")

    try:
        response = requests.get(args.image_url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image = np.array(image)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    print("Generating output...")

    output_text = sampler.chat(
        args.prompt, images=image, max_new_tokens=args.max_new_tokens
    )

    print("\n--- Generated Text ---")
    print(output_text)
    print("--------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma Inference with hfax")
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma_3_4b_it",
        help="Model ID (not used, but kept for compatibility).",
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
        default="What can you say about this image: <start_of_image>",
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
