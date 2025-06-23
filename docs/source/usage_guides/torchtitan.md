<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# TorchTitan

[TorchTitan](https://github.com/pytorch/torchtitan) is a large-scale LLM training library built with native PyTorch. It provides efficient distributed training for large language models with tensor parallelism, pipeline parallelism, data parallelism, and advanced memory optimization techniques.

TorchTitan implements several key features for large-scale model training:

1. **Multi-dimensional Parallelism**: Combines tensor parallelism, pipeline parallelism, and data parallelism
2. **Memory Optimization**: Advanced memory management and optimization strategies  
3. **Native PyTorch Integration**: Built using native PyTorch distributed primitives
4. **Flexible Configuration**: Highly configurable training setup via job config files
5. **Performance Optimization**: Optimized for large-scale training workloads
6. **Model Support**: Support for popular model architectures like Llama, Gemma, and more

TorchTitan is designed primarily for training large language models and provides the infrastructure needed to scale training across multiple nodes and GPUs efficiently.

Accelerate integrates [TorchTitan](https://github.com/pytorch/torchtitan) to provide seamless access to TorchTitan's distributed training capabilities within the familiar Accelerate workflow.

## What is integrated?

Training:

1. **TorchTitan Plugin Integration**: Accelerate provides a `TorchTitanPlugin` that allows you to configure and use TorchTitan's features within your existing Accelerate training scripts.

2. **Distributed Training Support**: Full integration with TorchTitan's multi-dimensional parallelism:
   - **Tensor Parallelism (TP)**: Distributes model weights across multiple GPUs
   - **Pipeline Parallelism (PP)**: Splits model layers across different GPUs
   - **Data Parallelism (DP)**: Replicates model across multiple devices for data parallelism
   - **Sequence Parallelism**: Distributes sequence computations across devices

3. **Memory Optimization**: Integration with TorchTitan's memory optimization features:
   - Gradient checkpointing integration
   - Memory-efficient attention mechanisms
   - Dynamic memory management

4. **Model Architecture Support**: Support for TorchTitan's optimized model implementations:
   - Llama model family (Llama2, Llama3, Code Llama)
   - Gemma models
   - Custom transformer architectures

## How it works?

**Pre-Requisites**: Install TorchTitan and its dependencies. Please refer to the [TorchTitan Installation guide](https://github.com/pytorch/torchtitan#installation) for more information.

### Accelerate TorchTitan Plugin

The easiest way to get started with TorchTitan in Accelerate is through the `TorchTitanPlugin`. You can configure TorchTitan in two ways:

#### Method 1: Environment Variable

Set the environment variable to enable TorchTitan:

```bash
export ACCELERATE_USE_TORCHTITAN=true
```

Then run your training script:

```bash
accelerate launch my_script.py --args_to_my_script
```

#### Method 2: TorchTitanPlugin in Code

```python
from accelerate import Accelerator
from accelerate.utils import TorchTitanPlugin

# Basic TorchTitan setup
torchtitan_plugin = TorchTitanPlugin(model_name="llama3")
accelerator = Accelerator(torchtitan_plugin=torchtitan_plugin)

# Your training code here
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

### TorchTitan Job Configuration

For more advanced configurations, you can use TorchTitan's job configuration system:

```python
from accelerate import Accelerator
from accelerate.utils import TorchTitanPlugin

# Advanced TorchTitan configuration
torchtitan_config = {
    "training": {
        "batch_size": 4,
        "seq_len": 2048,
        "gradient_accumulation_steps": 8,
        "max_steps": 10000,
        "learning_rate": 3e-4,
        "warmup_steps": 2000
    },
    "model": {
        "name": "llama3",
        "vocab_size": 32000,
        "embed_dim": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "intermediate_size": 11008
    },
    "parallelism": {
        "tensor_parallel": 2,
        "pipeline_parallel": 1,
        "data_parallel": 4,
        "sequence_parallel": False
    },
    "optimizer": {
        "name": "adamw",
        "lr": 3e-4,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8
    },
    "checkpointing": {
        "save_interval": 1000,
        "save_dir": "./checkpoints",
        "resume_from_checkpoint": null
    },
    "memory": {
        "gradient_checkpointing": True,
        "activation_checkpointing": True,
        "cpu_offload": False
    }
}

torchtitan_plugin = TorchTitanPlugin(
    job_config=torchtitan_config,
    model_name="llama3"
)

accelerator = Accelerator(
    torchtitan_plugin=torchtitan_plugin,
    mixed_precision="bf16",
    gradient_accumulation_steps=8
)
```

### Configuration from TOML File

TorchTitan also supports loading configuration from TOML files:

```python
from accelerate import Accelerator
from accelerate.utils import TorchTitanPlugin

# Load config from TOML file
torchtitan_plugin = TorchTitanPlugin(
    job_config="path/to/torchtitan_config.toml",
    model_name="llama3"
)

accelerator = Accelerator(torchtitan_plugin=torchtitan_plugin)
```

Example TOML configuration file (`torchtitan_config.toml`):

```toml
[training]
batch_size = 4
seq_len = 2048
gradient_accumulation_steps = 8
max_steps = 10000
learning_rate = 3e-4
warmup_steps = 2000

[model]
name = "llama3"
vocab_size = 32000
embed_dim = 4096
num_layers = 32
num_heads = 32
intermediate_size = 11008

[parallelism]
tensor_parallel = 2
pipeline_parallel = 1
data_parallel = 4
sequence_parallel = false

[optimizer]
name = "adamw"
lr = 3e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
eps = 1e-8

[checkpointing]
save_interval = 1000
save_dir = "./checkpoints"
resume_from_checkpoint = ""

[memory]
gradient_checkpointing = true
activation_checkpointing = true
cpu_offload = false
```

## Complete Training Example

Here's a complete example of using TorchTitan with Accelerate:

```python
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import TorchTitanPlugin
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    def __init__(self, size=1000, seq_len=512, vocab_size=32000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = torch.cat([input_ids[1:], torch.zeros(1, dtype=torch.long)])
        return {"input_ids": input_ids, "labels": labels}

def main():
    # TorchTitan configuration
    torchtitan_config = {
        "training": {
            "batch_size": 2,
            "seq_len": 512,
            "gradient_accumulation_steps": 4,
            "max_steps": 1000
        },
        "model": {
            "name": "llama3",
            "vocab_size": 32000,
            "embed_dim": 1024,
            "num_layers": 12,
            "num_heads": 16
        },
        "parallelism": {
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "data_parallel": 1
        }
    }
    
    # Initialize TorchTitan plugin
    torchtitan_plugin = TorchTitanPlugin(
        job_config=torchtitan_config,
        model_name="llama3"
    )
    
    # Initialize Accelerator with TorchTitan
    accelerator = Accelerator(
        torchtitan_plugin=torchtitan_plugin,
        mixed_precision="bf16",
        gradient_accumulation_steps=4
    )
    
    # Model, optimizer, and data
    from transformers import LlamaForCausalLM, LlamaConfig
    
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=1024,
        intermediate_size=2752,
        num_hidden_layers=12,
        num_attention_heads=16,
        max_position_embeddings=2048
    )
    
    model = LlamaForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    
    dataset = SimpleDataset(size=1000, seq_len=512)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    # Prepare everything with TorchTitan
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    # Training loop
    model.train()
    for step, batch in enumerate(dataloader):
        if step >= 100:  # Limit for example
            break
            
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        if step % 10 == 0:
            accelerator.print(f"Step {step}, Loss: {loss.item():.4f}")
    
    accelerator.print("Training completed!")

if __name__ == "__main__":
    main()
```

## Key Features and Configuration Options

### Parallelism Configuration

TorchTitan supports multiple parallelism strategies that can be combined:

```python
parallelism_config = {
    "tensor_parallel": 4,      # Split model weights across 4 GPUs
    "pipeline_parallel": 2,    # Split model layers across 2 stages  
    "data_parallel": 2,        # Replicate model across 2 data parallel groups
    "sequence_parallel": True  # Enable sequence parallelism for memory efficiency
}
```

### Memory Optimization

```python
memory_config = {
    "gradient_checkpointing": True,    # Save memory by recomputing activations
    "activation_checkpointing": True,  # Checkpoint specific layers
    "cpu_offload": False,             # Offload parameters to CPU
    "mixed_precision": "bf16"         # Use bfloat16 for memory efficiency
}
```

### Model-Specific Settings

```python
# Llama model configuration
llama_config = {
    "model": {
        "name": "llama3",
        "vocab_size": 32000,
        "embed_dim": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "intermediate_size": 11008,
        "max_seq_len": 2048,
        "rope_theta": 10000.0
    }
}

# Gemma model configuration  
gemma_config = {
    "model": {
        "name": "gemma",
        "vocab_size": 256000,
        "embed_dim": 3072,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 16,
        "intermediate_size": 24576
    }
}
```

## Environment Variables

TorchTitan integration supports several environment variables for configuration:

```bash
# Enable TorchTitan
export ACCELERATE_USE_TORCHTITAN=true

# TorchTitan specific settings
export TORCHTITAN_CONFIG_PATH="/path/to/config.toml"
export TORCHTITAN_MODEL_NAME="llama3"
export TORCHTITAN_CHECKPOINT_DIR="/path/to/checkpoints"

# Parallelism settings
export TORCHTITAN_TP_SIZE=2
export TORCHTITAN_PP_SIZE=1  
export TORCHTITAN_DP_SIZE=4

# Memory optimization
export TORCHTITAN_GRADIENT_CHECKPOINTING=true
export TORCHTITAN_ACTIVATION_CHECKPOINTING=true
```

## Multi-Node Training

TorchTitan works seamlessly with Accelerate's multi-node training:

```bash
# Node 0 (main node)
accelerate launch \
    --config_file accelerate_config.yaml \
    --main_process_ip 192.168.1.10 \
    --main_process_port 29500 \
    --machine_rank 0 \
    --num_machines 2 \
    --num_processes 16 \
    train_script.py

# Node 1
accelerate launch \
    --config_file accelerate_config.yaml \
    --main_process_ip 192.168.1.10 \
    --main_process_port 29500 \
    --machine_rank 1 \
    --num_machines 2 \
    --num_processes 16 \
    train_script.py
```

## Performance Tips

### 1. Optimal Parallelism Configuration

- **Tensor Parallelism**: Use when model layers are too large for single GPU memory
- **Pipeline Parallelism**: Effective for very deep models, but introduces bubble overhead
- **Data Parallelism**: Most efficient for scaling across many GPUs
- **Sequence Parallelism**: Helps with memory when using long sequences

### 2. Memory Optimization

```python
# Enable gradient checkpointing for memory savings
torchtitan_config["memory"]["gradient_checkpointing"] = True

# Use mixed precision training
accelerator = Accelerator(
    torchtitan_plugin=torchtitan_plugin,
    mixed_precision="bf16"  # or "fp16"
)

# Optimize batch size and gradient accumulation
torchtitan_config["training"]["batch_size"] = 1  # Micro batch size
torchtitan_config["training"]["gradient_accumulation_steps"] = 32  # Effective batch size
```

### 3. Communication Optimization

```python
# Optimize communication for tensor parallelism
torchtitan_config["parallelism"]["tensor_parallel_communication_backend"] = "nccl"

# Enable communication overlapping
torchtitan_config["training"]["overlap_communication"] = True
```

## Troubleshooting

### Common Issues and Solutions

1. **Out of Memory Errors**:
   ```python
   # Reduce micro batch size
   torchtitan_config["training"]["batch_size"] = 1
   
   # Increase gradient accumulation
   torchtitan_config["training"]["gradient_accumulation_steps"] = 64
   
   # Enable gradient checkpointing
   torchtitan_config["memory"]["gradient_checkpointing"] = True
   ```

2. **Slow Training Speed**:
   ```python
   # Optimize parallelism configuration
   torchtitan_config["parallelism"]["tensor_parallel"] = 2  # Reduce TP if too high
   torchtitan_config["parallelism"]["pipeline_parallel"] = 1  # Minimize PP overhead
   
   # Enable communication optimizations
   torchtitan_config["training"]["overlap_communication"] = True
   ```

3. **Checkpoint Issues**:
   ```python
   # Ensure checkpoint directory is accessible to all ranks
   torchtitan_config["checkpointing"]["save_dir"] = "/shared/checkpoints"
   
   # Set appropriate save intervals
   torchtitan_config["checkpointing"]["save_interval"] = 1000
   ```

### Debugging

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# TorchTitan will provide detailed logs about:
# - Model partitioning
# - Communication patterns  
# - Memory usage
# - Performance metrics
```

## Migration from Other Frameworks

### From DeepSpeed

Key differences when migrating from DeepSpeed to TorchTitan:

```python
# DeepSpeed style
from accelerate.utils import DeepSpeedPlugin
deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=3,
    offload_optimizer_device="cpu"
)

# TorchTitan equivalent
from accelerate.utils import TorchTitanPlugin
torchtitan_plugin = TorchTitanPlugin(
    job_config={
        "parallelism": {
            "tensor_parallel": 2,
            "data_parallel": 2
        },
        "memory": {
            "cpu_offload": True
        }
    }
)
```

### From FSDP

```python
# FSDP style
from accelerate.utils import FullyShardedDataParallelPlugin
fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy="FULL_SHARD",
    cpu_offload=True
)

# TorchTitan equivalent  
torchtitan_plugin = TorchTitanPlugin(
    job_config={
        "parallelism": {
            "tensor_parallel": 1,
            "data_parallel": 8  # Full data parallelism
        },
        "memory": {
            "cpu_offload": True
        }
    }
)
```

## Additional Resources

- [TorchTitan GitHub Repository](https://github.com/pytorch/torchtitan)
- [TorchTitan Documentation](https://github.com/pytorch/torchtitan/tree/main/docs)
- [TorchTitan Examples](https://github.com/pytorch/torchtitan/tree/main/train_configs)
- [Multi-dimensional Parallelism Paper](https://arxiv.org/abs/1909.08053)
- [Sequence Parallelism Techniques](https://arxiv.org/abs/2105.13120)

The TorchTitan integration in Accelerate provides a powerful and flexible way to scale your large language model training across multiple GPUs and nodes while maintaining the simplicity and ease of use that Accelerate is known for. 