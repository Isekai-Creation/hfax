#!/usr/bin/env python3
"""
Fine-tuning Gemma3 4B using the Kauldron library, based on the official Gemma documentation.
"""

import os
import optax
from kauldron import kd
import hfax
from transformers import AutoProcessor
from grain import python as grain

model_id = "unsloth/gemma-3-4b-it"

from PIL import Image  # Import PIL for image handling


class CustomImageTextToTokens(grain.MapTransform):
    def __init__(self, processor, tokenizer, max_length):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def map(self, features):
        # features will contain {'image': PIL.Image, 'text': str}

        # 1. Format into conversation (similar to convert_to_conversation)
        instruction = "Convert the equation images to LaTeX equations."
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": features["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": features["text"]}],
            },
        ]

        # 2. Process with HuggingFace processor
        # This part is crucial. The processor handles image processing and tokenization.
        batch = self.processor(
            text=[
                self.processor.apply_chat_template(
                    conversation, add_generation_prompt=False, tokenize=False
                )
            ],
            images=[features["image"].convert("RGB")],  # Ensure RGB format
            return_tensors="np",  # Return numpy arrays for JAX compatibility
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # 3. Prepare labels (similar to collate_fn)
        labels = batch["input_ids"].copy()
        labels[labels == 0] = -100
        # The image token ID might be different for Gemma3.
        # Need to get the actual image token ID from the processor's special tokens map.
        # For Gemma3, it's often 262144 or processor.tokenizer.special_tokens_map["boi_token"]
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.tokenizer.special_tokens_map.get("boi_token", "<image>")
        )
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100  # Hardcoded for now, but should be dynamic

        return {
            "input": batch["input_ids"],
            "target": labels.reshape(*labels.shape, 1),
            "loss_mask": (labels != -100).reshape(*labels.shape, 1),
        }


def main():
    """Main function to set up and run the training process."""

    # Set JAX to utilize the full GPU/TPU memory

    print("=== Gemma Kauldron Fine-tuning Setup ===")

    # 1. Create the tokenizer
    print("\n1. Initializing tokenizer...")
    tokenizer = hfax.text.Gemma3Tokenizer()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("✓ Tokenizer initialized.")

    # 2. Create the data pipeline
    print("\n2. Setting up data pipeline for dataset...")
    ds = kd.data.py.HuggingFace(
        path="unsloth/LaTeX_OCR",
        split="train[:3000]",
        shuffle=True,
        batch_size=8,
        num_epochs=5,
        num_workers=16,
        transforms=[
            CustomImageTextToTokens(
                processor=processor,
                tokenizer=tokenizer,
                max_length=4096,
            ),
        ],
        cache_dir="/dev/shm/dataset_cache",
    )
    print(f"len(ds): {len(ds)}")
    print("✓ Data pipeline created.")

    # Evaluation dataset
    eval_ds = kd.data.py.HuggingFace(
        path="unsloth/LaTeX_OCR",
        split="test[:300]",  # Use 'test' split
        shuffle=False,  # No shuffle for evaluation
        batch_size=8,  # Same batch size as train_ds
        num_epochs=1,  # Evaluate once per full pass if num_batches is None
        num_workers=16,
        transforms=[
            CustomImageTextToTokens(
                processor=processor,
                tokenizer=tokenizer,
                max_length=4096,
            ),
        ],
        cache_dir="/dev/shm/dataset_cache",
    )

    # 3. Define the model
    print("\n3. Defining Gemma 3 4B model...")
    model = hfax.nn.Gemma3_4B(
        tokens="batch.input",
    )
    print("✓ Model defined.")

    # 4. Define the loss function
    print("\n4. Defining loss function (SoftmaxCrossEntropy)...")
    loss = kd.losses.SoftmaxCrossEntropyWithIntLabels(
        logits="preds.logits",
        labels="batch.target",
        mask="batch.loss_mask",
    )
    print("✓ Loss function defined.")

    # 5. Create the trainer
    print("\n5. Configuring the Kauldron trainer...")
    trainer = kd.train.Trainer(
        seed=42,
        workdir="/dev/shm/kauldron_checkpoints_full",
        # Optional: Add checkpointing
        checkpointer=kd.ckpts.Checkpointer(
            save_interval_steps=600,
        ),
        # Dataset
        train_ds=ds,
        # evals
        evals={
            "eval": kd.evals.Evaluator(
                run=kd.evals.EveryNSteps(600),
                ds=eval_ds,
            )
        },
        # Model
        model=model,
        init_transform=hfax.ckpts.LoadCheckpoint(
            path=hfax.ckpts.CheckpointPath.GEMMA3_4B_IT,
        ),
        # Training parameters
        num_train_steps=len(ds),
        train_losses={"loss": loss},
        optimizer=optax.adafactor(learning_rate=1e-3),
    )
    print("✓ Trainer configured.")

    # 6. Start training
    print(f"\n6. Starting training for {len(ds)} steps...")
    # The state contains the trained parameters
    state, aux = trainer.train()
    print("\n✓ Training finished.")

    # 7. Perform evaluation
    print("\n7. Running a sample evaluation...")
    sampler = hfax.text.ChatSampler(
        model=model,
        params=state.params,
        tokenizer=tokenizer,
    )

    prompt = "Hello! My next holidays are in Paris."
    response = sampler.chat(prompt)

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")
    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()
