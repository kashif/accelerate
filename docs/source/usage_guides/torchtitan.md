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
2. **FSDP Integration**: Advanced Fully Sharded Data Parallelism support with flexible configurations
3. **Memory Optimization**: Advanced memory management including activation checkpointing and compilation
4. **Native PyTorch Integration**: Built using native PyTorch distributed primitives
5. **Flexible Configuration**: Highly configurable training setup via job config files or plugin parameters
6. **Advanced Checkpointing**: Comprehensive checkpoint management with async support and fault tolerance
7. **Model Support**: Support for popular model architectures like Llama, Gemma, and more

TorchTitan is designed primarily for training large language models and provides the infrastructure needed to scale training across multiple nodes and GPUs efficiently.

Accelerate integrates [TorchTitan](https://github.com/pytorch/torchtitan) to provide seamless access to TorchTitan's distributed training capabilities within the familiar Accelerate workflow.

## What is integrated?

Training:

1. **TorchTitan Plugin Integration**: Accelerate provides a comprehensive `TorchTitanPlugin` that allows you to configure and use TorchTitan's features within your existing Accelerate training scripts.

2. **Multi-dimensional Parallelism**: Full integration with TorchTitan's parallelism strategies:
   - **Tensor Parallelism (TP)**: Distributes model weights across multiple GPUs
   - **Pipeline Parallelism (PP)**: Splits model layers across different GPUs  
   - **Data Parallelism (DP)**: Replicates model across multiple devices for data parallelism
   - **FSDP Integration**: Advanced Fully Sharded Data Parallelism with configurable sharding strategies

3. **Memory Optimization**: Integration with TorchTitan's memory optimization features:
   - Activation checkpointing with selective layer targeting
   - Model compilation with torch.compile integration
   - Mixed precision training support
   - CPU offloading capabilities

4. **Advanced Checkpointing**: Comprehensive checkpoint management:
   - Asynchronous checkpointing for improved performance
   - Configurable checkpoint intervals and retention policies
   - Fault tolerance and recovery mechanisms
   - Support for loading partial checkpoints

5. **Model Architecture Support**: Support for TorchTitan's optimized model implementations:
   - Llama model family (Llama2, Llama3, Code Llama)
   - Gemma models
   - Custom transformer architectures

## How it works?

**Pre-Requisites**: Install TorchTitan and its dependencies. Please refer to the [TorchTitan Installation guide](https://github.com/pytorch/torchtitan#installation) for more information.

### Accelerate TorchTitan Plugin

The easiest way to get started with TorchTitan in Accelerate is through the `TorchTitanPlugin`. You can configure TorchTitan in several ways:

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
torchtitan_plugin = TorchTitanPlugin(
    model_name="llama3",
    tp_degree=2,
    pp_degree=1,
    enable_fsdp=True,
    activation_checkpointing=True
)
accelerator = Accelerator(torchtitan_plugin=torchtitan_plugin)

# Your training code here
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

### Comprehensive TorchTitan Configuration

For more advanced configurations, you can use TorchTitan's comprehensive configuration system:

```python
from accelerate import Accelerator
from accelerate.utils import TorchTitanPlugin

# Advanced TorchTitan configuration
torchtitan_plugin = TorchTitanPlugin(
    # Basic model configuration
    model_name="llama3",
    
    # Parallelism configuration
    tp_degree=4,           # Tensor parallelism degree
    pp_degree=2,           # Pipeline parallelism degree
    dp_degree=None,        # Auto-calculated data parallelism
    
    # FSDP configuration
    enable_fsdp=True,
    fsdp_sharding_strategy="FULL_SHARD",  # Options: "FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"
    fsdp_backward_prefetch="BACKWARD_PRE", # Options: "BACKWARD_PRE", "BACKWARD_POST", "NO_PREFETCH"
    fsdp_cpu_offload=False,
    fsdp_mixed_precision_policy={
        "param_dtype": "bfloat16",
        "reduce_dtype": "float32",
        "buffer_dtype": "float32"
    },
    
    # Memory optimization
    activation_checkpointing=True,
    selective_checkpointing_layers=["TransformerBlock"],  # Target specific layers
    
    # Model compilation
    compile_model=True,
    compile_config={
        "backend": "inductor",
        "mode": "reduce-overhead",
        "fullgraph": False,
        "dynamic": False
    },
    
    # Advanced checkpointing
    enable_checkpoint=True,
    checkpoint_folder="./torchtitan_checkpoints",
    checkpoint_interval=500,
    checkpoint_async_mode="async_with_pinned_mem",  # Options: "disabled", "async", "async_with_pinned_mem"
    checkpoint_keep_latest_k=3,
    checkpoint_export_dtype="float32",
    checkpoint_enable_first_step=True,
    checkpoint_model_weights_only_at_end=False,
    checkpoint_initial_load_path=None,
    
    # Job configuration (can also be a path to TOML file)
    job_config={
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
        }
    }
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
    model_name="llama3",
    tp_degree=2,
    enable_fsdp=True
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
tp_size = 2
pp_size = 1
dp_size = 4
enable_fsdp = true

[optimizer]
name = "adamw"
lr = 3e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
eps = 1e-8

[checkpointing]
enable = true
folder = "./torchtitan_checkpoints"
interval = 1000
async_mode = "async_with_pinned_mem"
keep_latest_k = 2
export_dtype = "float32"

[memory]
activation_checkpointing = true
selective_checkpointing_layers = ["TransformerBlock"]
compile_model = true
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
    torchtitan_plugin = TorchTitanPlugin(
        model_name="llama3",
        
        # Parallelism settings
        tp_degree=2,
        pp_degree=1,
        enable_fsdp=True,
        fsdp_sharding_strategy="FULL_SHARD",
        
        # Memory optimization
        activation_checkpointing=True,
        compile_model=True,
        
        # Checkpointing
        checkpoint_interval=100,
        checkpoint_async_mode="async",
        checkpoint_keep_latest_k=2,
        
        # Job configuration
        job_config={
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
            }
        }
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
    
    # Training loop with checkpointing
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
        
        # Automatic checkpointing based on plugin configuration
        if step % 50 == 0:
            accelerator.save_torchtitan_checkpoint(step)
        
        if step % 10 == 0:
            accelerator.print(f"Step {step}, Loss: {loss.item():.4f}")
    
    # Save final checkpoint
    accelerator.save_torchtitan_checkpoint(step, force=True)
    accelerator.print("Training completed!")

if __name__ == "__main__":
    main()
```

## Key Features and Configuration Options

### Parallelism Configuration

TorchTitan supports multiple parallelism strategies that can be combined:

```python
# Comprehensive parallelism setup
torchtitan_plugin = TorchTitanPlugin(
    tp_degree=4,           # Split model weights across 4 GPUs
    pp_degree=2,           # Split model layers across 2 pipeline stages  
    dp_degree=None,        # Auto-calculate: world_size / (tp_degree * pp_degree)
    
    # FSDP configuration
    enable_fsdp=True,
    fsdp_sharding_strategy="FULL_SHARD",    # Full parameter sharding
    fsdp_backward_prefetch="BACKWARD_PRE",  # Prefetch strategy
    fsdp_cpu_offload=False,                 # Keep parameters on GPU
    fsdp_mixed_precision_policy={
        "param_dtype": "bfloat16",
        "reduce_dtype": "float32",
        "buffer_dtype": "bfloat16"
    }
)
```

### Memory Optimization

```python
# Advanced memory optimization
torchtitan_plugin = TorchTitanPlugin(
    # Activation checkpointing
    activation_checkpointing=True,
    selective_checkpointing_layers=["TransformerBlock", "LlamaDecoderLayer"],
    
    # Model compilation
    compile_model=True,
    compile_config={
        "backend": "inductor",      # Compilation backend
        "mode": "reduce-overhead",  # Compilation mode
        "fullgraph": False,         # Allow graph breaks
        "dynamic": False            # Static shapes
    },
    
    # FSDP CPU offloading
    fsdp_cpu_offload=True
)
```

### Advanced Checkpointing

```python
# Comprehensive checkpointing configuration
torchtitan_plugin = TorchTitanPlugin(
    enable_checkpoint=True,
    checkpoint_folder="./distributed_checkpoints",
    checkpoint_interval=500,
    
    # Async checkpointing for performance
    checkpoint_async_mode="async_with_pinned_mem",  # Options: "disabled", "async", "async_with_pinned_mem"
    
    # Checkpoint management
    checkpoint_keep_latest_k=3,                     # Keep 3 latest checkpoints
    checkpoint_enable_first_step=True,              # Save checkpoint at step 0
    checkpoint_export_dtype="float32",              # Export precision
    checkpoint_model_weights_only_at_end=False,     # Save full state at end
    
    # Loading configuration
    checkpoint_initial_load_path="./pretrained_checkpoint",
    checkpoint_initial_load_model_weights_only=False,
    checkpoint_exclude_from_loading=["optimizer_states"]
)

# Manual checkpointing in training loop
accelerator = Accelerator(torchtitan_plugin=torchtitan_plugin)

# In training loop
for step, batch in enumerate(dataloader):
    # ... training code ...
    
    # Manual checkpoint save
    if step % 1000 == 0:
        accelerator.save_torchtitan_checkpoint(step, force=True)
    
    # Load checkpoint if needed
    if should_resume:
        loaded = accelerator.load_torchtitan_checkpoint(step=-1)  # Load latest
        if loaded:
            accelerator.print(f"Resumed from step {accelerator.get_torchtitan_checkpoint_step()}")
```

## Environment Variables

TorchTitan integration supports several environment variables for configuration:

```bash
# Enable TorchTitan
export ACCELERATE_USE_TORCHTITAN=true

# Parallelism settings
export TORCHTITAN_TP_DEGREE=2
export TORCHTITAN_PP_DEGREE=1
export TORCHTITAN_DP_DEGREE=4  # 0 means auto-calculate

# FSDP settings
export TORCHTITAN_ENABLE_FSDP=true

# Memory optimization
export TORCHTITAN_ACTIVATION_CHECKPOINTING=true
export TORCHTITAN_COMPILE_MODEL=true

# Other TorchTitan specific settings
export TORCHTITAN_CONFIG_PATH="/path/to/config.toml"
export TORCHTITAN_MODEL_NAME="llama3"
export TORCHTITAN_CHECKPOINT_DIR="/path/to/checkpoints"
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
- **FSDP**: Combines benefits of data parallelism with memory efficiency

### 2. Memory Optimization

```python
# Enable comprehensive memory optimizations
torchtitan_plugin = TorchTitanPlugin(
    # Activation checkpointing for memory savings
    activation_checkpointing=True,
    selective_checkpointing_layers=["TransformerBlock"],
    
    # FSDP with CPU offloading
    enable_fsdp=True,
    fsdp_cpu_offload=True,
    fsdp_sharding_strategy="FULL_SHARD",
    
    # Model compilation for efficiency
    compile_model=True,
    compile_config={"backend": "inductor", "mode": "reduce-overhead"}
)

# Use mixed precision training
accelerator = Accelerator(
    torchtitan_plugin=torchtitan_plugin,
    mixed_precision="bf16"  # or "fp16"
)

# Optimize batch size and gradient accumulation
job_config = {
    "training": {
        "batch_size": 1,                    # Micro batch size
        "gradient_accumulation_steps": 32   # Effective batch size = 32
    }
}
```

### 3. Checkpointing Optimization

```python
# Async checkpointing for better performance
torchtitan_plugin = TorchTitanPlugin(
    checkpoint_async_mode="async_with_pinned_mem",  # Best performance
    checkpoint_interval=1000,                       # Reasonable frequency
    checkpoint_keep_latest_k=2,                     # Limit storage usage
    checkpoint_export_dtype="bfloat16"              # Reduce checkpoint size
)
```

## Troubleshooting

### Common Issues and Solutions

1. **Out of Memory Errors**:
   ```python
   # Reduce micro batch size
   job_config["training"]["batch_size"] = 1
   
   # Increase gradient accumulation
   job_config["training"]["gradient_accumulation_steps"] = 64
   
   # Enable memory optimizations
   torchtitan_plugin = TorchTitanPlugin(
       activation_checkpointing=True,
       fsdp_cpu_offload=True,
       enable_fsdp=True
   )
   ```

2. **Slow Training Speed**:
   ```python
   # Optimize parallelism configuration
   torchtitan_plugin = TorchTitanPlugin(
       tp_degree=2,    # Reduce TP if too high
       pp_degree=1,    # Minimize PP overhead
       enable_fsdp=True,
       
       # Enable compilation
       compile_model=True,
       
       # Use async checkpointing
       checkpoint_async_mode="async_with_pinned_mem"
   )
   ```

3. **Checkpoint Issues**:
   ```python
   # Ensure checkpoint directory is accessible to all ranks
   torchtitan_plugin = TorchTitanPlugin(
       checkpoint_folder="/shared/checkpoints",
       checkpoint_async_mode="async",
       checkpoint_keep_latest_k=2
   )
   ```

### World Size Validation

TorchTitan automatically validates that your parallelism configuration is compatible with the available world size:

```python
# Example: 8 GPUs total
torchtitan_plugin = TorchTitanPlugin(
    tp_degree=2,    # 2 GPUs for tensor parallelism
    pp_degree=1,    # 1 GPU for pipeline parallelism
    dp_degree=None  # Auto-calculated: 8 / (2 * 1) = 4
)

# Validation occurs automatically
accelerator = Accelerator(torchtitan_plugin=torchtitan_plugin)
```

### Debugging

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# TorchTitan will provide detailed logs about:
# - Model partitioning and device mesh setup
# - FSDP sharding strategies
# - Checkpoint operations and async handling
# - Memory usage and optimization
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
    enable_fsdp=True,
    fsdp_sharding_strategy="FULL_SHARD",
    fsdp_cpu_offload=True
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

# TorchTitan equivalent with additional features
torchtitan_plugin = TorchTitanPlugin(
    enable_fsdp=True,
    fsdp_sharding_strategy="FULL_SHARD",
    fsdp_cpu_offload=True,
    
    # Additional TorchTitan features
    tp_degree=2,                    # Tensor parallelism
    activation_checkpointing=True,  # Memory optimization
    compile_model=True              # Performance optimization
)
```

## Additional Resources

- [TorchTitan GitHub Repository](https://github.com/pytorch/torchtitan)
- [TorchTitan Documentation](https://github.com/pytorch/torchtitan/tree/main/docs)
- [TorchTitan Examples](https://github.com/pytorch/torchtitan/tree/main/train_configs)
- [Multi-dimensional Parallelism Paper](https://arxiv.org/abs/1909.08053)
- [FSDP Paper](https://arxiv.org/abs/2304.11277)

The TorchTitan integration in Accelerate provides a powerful and flexible way to scale your large language model training across multiple GPUs and nodes while maintaining the simplicity and ease of use that Accelerate is known for. With comprehensive checkpointing, advanced memory optimization, and multi-dimensional parallelism support, TorchTitan enables efficient training of the largest language models. 