# **Project TTPU: A Universal Library for Transformers on TPUs**

This document presents the expanded vision for TTPU, a production-ready, distributed-first library for running any Hugging Face transformers model on TPUs. It incorporates strategies for distributed execution, workload-specific optimizations, and seamless integration with the existing ML ecosystem.

## **1\. Proposal: Rebrand to TTPU**

To better reflect the project's ambitious goal, we propose rebranding from the internal codename hfax to **TTPU (Transformers on TPU)**. This name is clear, descriptive, and memorable. The standard import would become from ttpu import ....

## **2\. Expanded Project Goal**

The goal of TTPU is to be the definitive, easy-to-use solution for training and inference with Hugging Face models on TPUs. It will abstract away the complexities of model adaptation and distributed computing, enabling any ML engineer to leverage the full power of TPUs.

## **3\. Core Philosophy: Bridging the Framework Gap**

We recognize the fundamental tradeoff for TPU users today:

- **PyTorch with torch_xla**: Offers excellent compatibility with the Hugging Face ecosystem. A user can often just specify device="xla" and move a model to the TPU. However, achieving peak performance can be challenging and requires deep expertise.
- **JAX/Flax**: Offers incredible performance and scalability on TPUs due to its design as a JIT-compiled functional framework. However, it lacks native compatibility with the vast number of PyTorch-native models on the Hub.

**TTPU's mission is to bridge this gap.** We will provide a unified interface that delivers the **simplicity of torch_xla** with the **performance of native JAX**, allowing users to choose the backend that best fits their needs without compromising on usability or speed.

## **4\. Architectural Pillars**

The TTPU library will be built upon five key architectural pillars.

### **Pillar 1: Intelligent Model Adaptation**

This is the foundational layer, responsible for making models compatible with TPU backends.

- **Configuration-Driven Adaptation:** How can we leverage each model's config.json file to programmatically understand its architecture (e.g., number of layers, attention type, activation functions) and automatically apply the correct, pre-defined set of TPU-optimized module replacements?
- **Module Replacement Strategy:** Instead of complex patching, our primary strategy will be to replace entire modules (e.g., LlamaAttention, MistralMLP) with TPU-optimized equivalents that have identical APIs. This ensures both performance and maintainability.
- **Handling Architectural Complexity:**
  - How do we create a robust system of "standard adapters" for common architectures (Llama, Mistral, etc.)?
  - For highly custom or multimodal models (e.g., Gemma 3, Qwen-VL), how do we design a plug-in system, similar to transformers/models/model_name, that allows TTPU to use a dedicated, model-specific adapter for full compatibility?

### **Pillar 2: Natively Distributed Execution**

TTPU will be "distributed-first," with the **SPMD (Single Program, Multiple Data) paradigm as the default mode** for both training and inference.

- **JAX Backend (SPMD Focus):**
  - **Automatic Sharding:** How can TTPU abstract away JAX's sharding primitives? The library should automatically infer and apply optimal **SPMD** sharding strategies (tensor and pipeline parallelism) based on the model architecture and TPU topology, without the user ever needing to define a Mesh or PartitionSpec.
  - **Simplified Process Management:** How do we provide simple utilities (e.g., ttpu launch) to manage the multi-host, multi-process setup required by JAX on TPU Pods?
- **PyTorch / torch_xla Backend (FSDP & Multi-Process Focus):**
  - **FSDP Abstraction:** How do we wrap FSDP so that a user can enable it with a single, high-level parameter (e.g., parallelism_strategy='fsdp'), while TTPU handles the complex wrapping policies and mesh creation under the hood?
  - **Simplified Process Spawning:** How will TTPU wrap and simplify the torch_xla.launch script, abstracting away the need for users to manage the multi-process environment manually?

### **Pillar 3: Workload-Specific Optimizations**

- **Training Optimizations:**
  - **Optimizer State Sharding:** How will TTPU integrate and manage sharded optimizers to achieve **ZeRO stages 2 and 3** memory savings for both backends?
  - **Efficient Gradient Communication:** How do we ensure TTPU utilizes the most efficient all-reduce and collective communication primitives on the TPU's high-speed interconnect?
- **Inference Optimizations:**
  - **Distributed KV Cache:** The initial focus will be on a **statically sized KV cache**, which is highly amenable to efficient sharding on TPUs. How will TTPU automatically calculate the cache size and distribute it across all available TPU memory?
  - **High-Throughput Serving:** How can TTPU become the **vLLM-equivalent for TPUs**?
    - **Continuous Batching (R\&D Goal):** What is the path to implementing a TPU-native version of **PagedAttention** to enable a high-throughput, continuous batching request scheduler?
    - **Speculative Decoding:** What support will TTPU provide for configuring and running speculative decoding in a distributed TPU environment?

### **Pillar 4: Seamless Ecosystem Integration**

- **Hugging Face Accelerate:** Should TTPU be a **backend for Accelerate**? Can a user simply select a TTPU device in their Accelerate config to unlock all its features? How will TTPU behave when launched via accelerate launch?
- **DeepSpeed & vLLM Feature Parity:** The goal is not direct integration, but **feature parity**. How can TTPU provide a DeepSpeed\-like configuration experience for training and a vLLM\-like engine for inference, tailored specifically for the TPU architecture?

### **Pillar 5: A World-Class Developer Experience**

- **The Unified API:** All complexity should be hidden behind a simple, powerful API.  
  from ttpu import AutoModelForTPU

  \# High-performance inference is now a one-liner.  
  \# TTPU handles sharding, module replacement, and device placement.  
  model \= AutoModelForTPU.from_pretrained(  
   "google/gemma-3-27b-it",  
   backend='jax',  
   task='inference',  
   parallelism_strategy='auto',  
  )

  \# Distributed training is just as simple.  
  trainer \= ttpu.Trainer(  
   model="meta-llama/Meta-Llama-3-70B",  
   backend='torch_xla',  
   parallelism_strategy='fsdp',  
   zero_stage=3,  
   \# ... trainer args  
  )

- **Intelligent Defaults:** What are the most effective default settings we can provide so that a user gets a powerful, distributed setup out of the box with minimal configuration?
- **Clear Error Messaging:** If a model is not yet supported or a specific configuration is invalid, how do we provide clear, actionable error messages that guide the user to a solution?

## **5\. Future Directions**

While the above pillars form our core roadmap, we will also explore:

- **Advanced Quantization:** Integration of TPU-native quantization techniques like AQT.
- **Multi-Model Serving:** Supporting the deployment of multiple, independent models on a single TPU pod to maximize hardware utilization, managed via a TTPU server.
- **Experiment Tracking:** Seamless integration with tools like Weights & Biases or TensorBoard.
