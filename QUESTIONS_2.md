# **Project TTPU: A Universal Library for Transformers on TPUs**

This document presents the expanded vision for TTPU, a production-ready, distributed-first library for running any Hugging Face transformers model on TPUs. It incorporates strategies for distributed execution, workload-specific optimizations, and seamless integration with the existing ML ecosystem.

## **1\. Proposal: Rebrand to TTPU**

To better reflect the project's ambitious goal, we propose rebranding from the internal codename hfax to **TTPU (Transformers on TPU)**. This name is clear, descriptive, and memorable. The standard import would become from ttpu import ....

## **2\. Expanded Project Goal**

The goal of TTPU is to be the definitive, easy-to-use solution for training and inference with Hugging Face models on TPUs. It will abstract away the complexities of model adaptation and distributed computing, enabling any ML engineer to leverage the full power of TPUs.

## **3\. Architectural Pillars**

The TTPU library will be built upon four key architectural pillars.

### **Pillar 1: Core Model Adaptation**

This is the foundational layer, responsible for making individual models compatible with TPU backends. It addresses the challenges of patching PyTorch code for torch_xla and bridging the gap to JAX/Flax, as detailed in our [Core Technical Strategy document](https://www.google.com/search?q=./README-Core-Strategy.md).

### **Pillar 2: Distributed Execution & Parallelism**

Large models require distributed execution. TTPU will be "distributed-first," providing powerful and simple abstractions for parallelism.

- **JAX Backend (SPMD Focus):**
  - **Automatic Sharding:** How can TTPU automatically apply optimal **SPMD** sharding strategies (e.g., tensor and pipeline parallelism) to a model's weights and activations without requiring manual user configuration?
  - **Mesh & Process Management:** How do we provide simple utilities (e.g., ttpu launch) to manage the multi-host, multi-process setup required by JAX on TPU Pods?
- **PyTorch / torch_xla Backend (FSDP & Multi-Process Focus):**
  - **FSDP Abstraction:** How can TTPU provide a dead-simple, high-level API to configure **Fully Sharded Data Parallelism (FSDP)**? A user should be able to enable it with a single parameter.
  - **Process Spawning:** How will TTPU wrap and simplify the torch_xla.launch script, abstracting away the need for users to manage the multi-process environment manually?
  - **Hybrid Parallelism:** What is the strategy for supporting more advanced hybrid topologies, such as combining FSDP with tensor parallelism, on the torch_xla backend?

### **Pillar 3: Differentiated Workloads (Inference vs. Training)**

The optimizations for training and inference are fundamentally different. TTPU will have specialized, high-performance paths for each.

- **Training Optimizations:**
  - **Optimizer State Sharding:** How will TTPU integrate and manage sharded optimizers to achieve **ZeRO stages 2 and 3** memory savings for both backends?
  - **Efficient Gradient Communication:** How do we ensure that TTPU utilizes the most efficient all-reduce and collective communication primitives available on the TPU's high-speed interconnect?
- **Inference Optimizations:**
  - **Distributed KV Cache:** How will TTPU implement automatic sharding of the Key-Value cache across all available TPU memory to support massive contexts and batch sizes?
  - **High-Throughput Serving:** How can TTPU become the **vLLM-equivalent for TPUs**? This requires addressing key performance technologies:
    - **Continuous Batching:** Can we implement a TPU-native version of **PagedAttention** to eliminate memory fragmentation and build a high-throughput request scheduler?
    - **Speculative Decoding:** What support will TTPU provide for configuring and running speculative decoding in a distributed TPU environment?

### **Pillar 4: Ecosystem Integration & Interoperability**

TTPU must integrate seamlessly with the tools developers already know and use.

- **Hugging Face Accelerate:**
  - Should TTPU be a **backend for Accelerate**? Can a user simply select a TTPU device in their Accelerate config to unlock all its features?
  - How will TTPU behave when launched via accelerate launch?
- **DeepSpeed & vLLM Feature Parity:**
  - The goal is not direct integration, but **feature parity**. How can TTPU provide a DeepSpeed-like configuration experience for training and a vLLM-like engine for inference, tailored specifically for the TPU architecture?

## **4\. The Unified TTPU API**

All of this complexity should be hidden behind a simple, powerful API.

- **The AutoModelForTPU Class:** How should the API evolve to support these advanced features? Consider the following conceptual example:  
  from ttpu import AutoModelForTPU

  \# High-performance inference setup  
  model \= AutoModelForTPU.from_pretrained(  
   "google/gemma-3-27b-it",  
   backend='jax',  
   task='inference',  
   parallelism_strategy='auto', \# TTPU figures out the best sharding  
   enable_paged_attention=True,  
  )

  \# Distributed training setup  
  trainer \= ttpu.Trainer(  
   model="meta-llama/Meta-Llama-3-70B",  
   backend='torch_xla',  
   parallelism_strategy='fsdp',  
   zero_stage=3,  
   ...  
  )

- **Intelligent Defaults:** What are the most effective and performant default settings we can provide, so that a user gets a powerful, distributed setup right out of the box with minimal configuration?
