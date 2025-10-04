import os
import jax.numpy as jnp
import tensorflow_datasets as tfds
import hfax as gm

def main():
    """
    This script demonstrates how to use the hfax library for multi-modal inference.
    """
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

    # First, let’s load an image:
    print("Loading image from oxford_flowers102 dataset...")
    ds = tfds.data_source('oxford_flowers102', split='train')
    image = ds[0]['image']
    print("Image loaded.")

    # Load the model and params.
    print("Loading Gemma3_4B model and parameters...")
    model = gm.nn.Gemma3_4B()
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)
    print("Model and parameters loaded.")

    # Sampling full prompt
    print("Creating ChatSampler...")
    sampler = gm.text.ChatSampler(
        model=model,
        params=params,
    )
    print("ChatSampler created.")

    print("Running inference...")
    out = sampler.chat(
        'What can you say about this image: \n',
        images=image,
    )
    print("\nOutput from model:")
    print(out)

if __name__ == '__main__':
    main()

