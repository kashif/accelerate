# Manual Gradient Accumulation Control with DeepSpeed

DeepSpeed handles gradient accumulation differently from standard PyTorch distributed training. While Accelerate provides automatic gradient accumulation through the `accumulate()` context manager, there are cases where you need fine-grained control over when DeepSpeed performs optimizer steps, gradient clipping, and other end-of-accumulation operations.

## The Problem

DeepSpeed's training loop is fundamentally different from standard PyTorch training because it automatically manages gradient accumulation based on the configuration parameters:

- `train_batch_size`: The effective global batch size
- `train_micro_batch_size_per_gpu`: Batch size processed by one GPU in one step  
- `gradient_accumulation_steps`: Number of micro-batches to accumulate before optimizer step

The relationship is: `train_batch_size = train_micro_batch_size_per_gpu × gradient_accumulation_steps × num_gpus`

In DeepSpeed's standard training loop, the `engine.step()` method is called after every `engine.backward()`, but internally DeepSpeed only performs the actual optimizer step, gradient reduction, and gradient clipping when it reaches the gradient accumulation boundary (determined by `gradient_accumulation_steps`).

However, in some advanced scenarios, users need explicit control over when these operations occur, independent of the configured `gradient_accumulation_steps`. Previously, users had to write conditional code to handle DeepSpeed's gradient accumulation boundaries:

```python
def backward(loss, is_final_micro_batch=False):
    """Perform backward pass with appropriate gradient accumulation boundary"""
    if using_deepspeed:
        # Tell DeepSpeed whether this is a boundary for gradient accumulation
        model.set_gradient_accumulation_boundary(is_final_micro_batch)
        # DeepSpeed's backward
        model.backward(loss)
    else:
        # accelerator's backward
        get_accelerator().backward(loss)
```

## The Solution

Accelerate now provides native support for DeepSpeed's gradient accumulation boundary control through:

1. **Enhanced `Accelerator.backward()` method** with an `is_final_micro_batch` parameter
2. **New `set_deepspeed_gradient_accumulation_boundary()` method** for explicit boundary control

### Method 1: Using `backward()` with `is_final_micro_batch`

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Manual gradient accumulation with explicit boundary control
for step, batch in enumerate(dataloader):
    for micro_batch_idx, micro_batch in enumerate(micro_batches):
        outputs = model(micro_batch)
        loss = outputs.loss
        
        # Explicitly specify when this is the final micro-batch
        is_final = (micro_batch_idx == len(micro_batches) - 1)
        accelerator.backward(loss, is_final_micro_batch=is_final)
```

### Method 2: Using explicit boundary control

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Alternative approach with explicit boundary setting
for step, batch in enumerate(dataloader):
    for micro_batch_idx, micro_batch in enumerate(micro_batches):
        # Set boundary before backward pass
        is_final = (micro_batch_idx == len(micro_batches) - 1)
        accelerator.set_deepspeed_gradient_accumulation_boundary(is_final)
        
        outputs = model(micro_batch)
        loss = outputs.loss
        accelerator.backward(loss)
```

### Method 3: Using `optimizer_step_and_zero_grad()` (Recommended)

For the most convenient experience, especially when you also need to handle optimizer stepping and gradient clipping, use the unified optimizer step method:

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Unified approach that works with both DeepSpeed and standard training
for batch in dataloader:
    with accelerator.accumulate(model):
        outputs = model(batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        if accelerator.sync_gradients:
            # This handles both DeepSpeed and standard training automatically
            grad_norm = accelerator.optimizer_step_and_zero_grad(
                optimizer=optimizer, 
                model=model, 
                max_grad_norm=1.0  # Optional gradient clipping
            )
            
            # grad_norm contains the gradient norm for monitoring
            if grad_norm is not None and grad_norm != -1.0:
                print(f"Gradient norm: {grad_norm}")
```

This method automatically:
- Sets the gradient accumulation boundary for DeepSpeed (when `sync_gradients=True`)
- Handles gradient clipping for both DeepSpeed and standard training
- Performs optimizer step and zero_grad operations
- Returns gradient norm for monitoring

## Key Features

### Backward Compatibility
- **Existing code continues to work unchanged**
- When `is_final_micro_batch` is not specified, Accelerate falls back to using `sync_gradients` (existing behavior)
- Non-DeepSpeed training is unaffected

### Automatic Fallback
```python
# This still works exactly as before
accelerator.backward(loss)  # Uses sync_gradients for boundary detection
```

### Fine-grained Control
```python
# Now you can explicitly control boundaries
accelerator.backward(loss, is_final_micro_batch=True)   # Forces boundary
accelerator.backward(loss, is_final_micro_batch=False)  # Prevents boundary
```

## How It Works

### DeepSpeed Integration
The enhanced `DeepSpeedEngineWrapper` now:

1. **Tracks gradient accumulation state** via `_gradient_accumulation_boundary` flag
2. **Conditionally calls `engine.step()`** only when at accumulation boundaries
3. **Forwards boundary information** to DeepSpeed's native `set_gradient_accumulation_boundary()` method (if available)

### Accelerator Changes
The `Accelerator.backward()` method now:

1. **Accepts `is_final_micro_batch` parameter** for explicit boundary control
2. **Falls back to `sync_gradients`** when the parameter is not provided
3. **Passes boundary information** to the DeepSpeed engine wrapper

## Use Cases

### Custom Gradient Accumulation Schedules
```python
# Variable micro-batch sizes or custom accumulation patterns
for step, batch in enumerate(dataloader):
    micro_batches = custom_split_batch(batch)  # Custom splitting logic
    
    for i, micro_batch in enumerate(micro_batches):
        is_final = (i == len(micro_batches) - 1)
        loss = model(micro_batch).loss
        accelerator.backward(loss, is_final_micro_batch=is_final)
```

### Dynamic Accumulation Steps
```python
# Changing accumulation steps during training
for step, batch in enumerate(dataloader):
    # Dynamically determine accumulation steps
    current_accumulation_steps = get_dynamic_accumulation_steps(step)
    
    for i in range(current_accumulation_steps):
        micro_batch = get_micro_batch(batch, i)
        is_final = (i == current_accumulation_steps - 1)
        
        loss = model(micro_batch).loss
        accelerator.backward(loss, is_final_micro_batch=is_final)
```

### Integration with Custom Training Loops
```python
# Advanced training scenarios with custom logic
def custom_training_step(accelerator, model, batch):
    # Complex micro-batching logic
    micro_batches = advanced_batch_processing(batch)
    
    total_loss = 0
    for idx, micro_batch in enumerate(micro_batches):
        outputs = model(micro_batch)
        loss = outputs.loss
        
        # Custom logic to determine boundaries
        is_boundary = should_step_optimizer(idx, micro_batch, outputs)
        accelerator.backward(loss, is_final_micro_batch=is_boundary)
        
        total_loss += loss.detach()
    
    return total_loss / len(micro_batches)
```

## Important Notes

### DeepSpeed Configuration
- The new functionality works with **any DeepSpeed ZeRO stage** (Stage 0, 1, 2, 3)
- **Gradient accumulation steps in DeepSpeed config** can still be used alongside manual control
- **Pipeline parallelism** requires special consideration (use DeepSpeed's native APIs)

### Performance Considerations
- Manual boundary control should be used **only when necessary**
- Frequent boundary changes may impact performance
- **Test thoroughly** with your specific use case

### Compatibility
- Requires **DeepSpeed 0.6.0 or later** for full functionality
- Works with **all Accelerate-supported DeepSpeed features**
- **Fallback behavior** ensures compatibility with older DeepSpeed versions

## Migration Guide

### Before (Manual Conditional Code)
```python
def backward_with_accumulation(accelerator, model, loss, is_final_micro_batch):
    if accelerator.distributed_type == "DEEPSPEED":
        model.set_gradient_accumulation_boundary(is_final_micro_batch)
        model.backward(loss)
    else:
        if is_final_micro_batch:
            accelerator.backward(loss)
        else:
            with accelerator.no_sync(model):
                accelerator.backward(loss)
```

### After (Native Accelerate Support)
```python
# Clean, unified API across all backends
accelerator.backward(loss, is_final_micro_batch=is_final_micro_batch)
```

This enhancement makes DeepSpeed's gradient accumulation control a first-class feature in Accelerate, eliminating the need for backend-specific conditional code while maintaining full backward compatibility. 