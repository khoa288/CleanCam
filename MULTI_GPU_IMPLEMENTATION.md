# Multi-GPU Training Implementation

## Summary

Successfully implemented multi-GPU training support using PyTorch's DataParallel. The pipeline now automatically detects and uses all available GPUs.

## Implementation Details

### Changes Made

1. **Training Module** (`cleancam_pipeline/models/training.py`):
   - Added DataParallel wrapper when multiple GPUs are detected
   - Updated print statements to show multi-GPU status
   - Handles model state dict correctly with DataParallel

2. **Configuration** (`cleancam_pipeline/core/config.py`):
   - Added `use_multi_gpu: bool = True` field
   - Updated `from_args()` to handle `--single-gpu` flag

3. **CLI** (`cleancam_pipeline.py`):
   - Added `--single-gpu` argument to disable multi-GPU training

4. **Benchmark Runner** (`cleancam_pipeline/orchestrators/benchmark_runner.py`):
   - Passes `use_multi_gpu` flag to training runs

5. **Documentation** (`README.md`):
   - Added Multi-GPU Training section
   - Updated performance tips

## How It Works

### Automatic Detection

```python
if torch.cuda.is_available() and torch.cuda.device_count() > 1 and cfg.use_multi_gpu:
    model = nn.DataParallel(model)
```

### DataParallel Behavior

- Automatically splits each batch across available GPUs
- Each GPU processes a portion of the batch in parallel
- Gradients are gathered and averaged across GPUs
- Single-process, multi-thread approach

### Example Output

**With 4 GPUs:**
```
[RunStart] model=resnet18 setting=train_real_only__eval_real_only seed=42 device=cuda (multi-GPU) ...
[MultiGPU] Using 4 GPUs with DataParallel
```

**With 1 GPU:**
```
[RunStart] model=resnet18 setting=train_real_only__eval_real_only seed=42 device=cuda ...
```

## Usage

### Default (Multi-GPU Enabled)

```bash
python cleancam_pipeline.py \
    --release-root /path/to/data \
    --output-root ./output \
    --run-benchmark \
    --models resnet18
```

If 4 GPUs are available, all 4 will be used automatically.

### Force Single GPU

```bash
python cleancam_pipeline.py \
    --release-root /path/to/data \
    --output-root ./output \
    --run-benchmark \
    --models resnet18 \
    --single-gpu
```

Only uses GPU 0, even if more are available.

### Force CPU

```bash
python cleancam_pipeline.py \
    --release-root /path/to/data \
    --output-root ./output \
    --run-benchmark \
    --models resnet18 \
    --cpu-only
```

## Performance Considerations

### Batch Size

With DataParallel, the effective batch size per GPU is:
```
batch_size_per_gpu = total_batch_size / num_gpus
```

Example with `--batch-size 256` and 4 GPUs:
- Each GPU processes 64 samples
- Total effective batch size: 256

### Scaling Efficiency

DataParallel scaling efficiency:
- 2 GPUs: ~1.8x speedup
- 4 GPUs: ~3.2x speedup
- 8 GPUs: ~5.5x speedup

Not perfectly linear due to:
- Communication overhead
- Python GIL (Global Interpreter Lock)
- Gradient synchronization

### When to Use Single GPU

Use `--single-gpu` when:
- Debugging model issues
- Batch size is too small (< 32 per GPU)
- Memory is limited
- Comparing with single-GPU baselines

## Limitations

### DataParallel Limitations

1. **Single Process**: All GPUs share Python GIL
2. **GPU 0 Bottleneck**: GPU 0 does extra work (gathering, loss computation)
3. **Scaling**: Best for 2-4 GPUs, diminishing returns beyond that

### Alternative: DistributedDataParallel

For better scaling (4+ GPUs), consider implementing DistributedDataParallel:
- Multi-process (no GIL bottleneck)
- Better load balancing
- Near-linear scaling
- More complex to implement

## Troubleshooting

### Check GPU Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### CUDA Out of Memory

If you get OOM errors with multi-GPU:

```bash
# Reduce batch size
--batch-size 128

# Or use single GPU
--single-gpu

# Or use CPU
--cpu-only
```

### Uneven GPU Utilization

This is normal with DataParallel - GPU 0 typically has higher utilization due to:
- Gathering outputs from other GPUs
- Computing final loss
- Scattering inputs to other GPUs

### Device Mismatch Errors

If you see "tensors on different devices" errors:
- Check that all inputs are moved to the correct device
- Ensure loss functions handle device placement correctly
- Verify custom layers don't hardcode device placement

## Testing

### Test Multi-GPU Setup

```python
import torch
import torch.nn as nn

# Check GPUs
print(f"GPUs available: {torch.cuda.device_count()}")

# Create simple model
model = nn.Linear(10, 5)

# Wrap with DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

model = model.cuda()

# Test forward pass
x = torch.randn(32, 10).cuda()
y = model(x)
print(f"Output shape: {y.shape}")
print("✓ Multi-GPU test passed!")
```

## Compatibility

### Supported

- ✅ All base models (ResNet18, MobileNetV2, EfficientNet-B0)
- ✅ Standard cross-entropy loss
- ✅ CORAL ordinal regression
- ✅ CORN ordinal regression
- ✅ Mixed precision (AMP)
- ✅ Weighted sampling
- ✅ Class weights

### Requirements

- PyTorch >= 1.10.0
- CUDA-capable GPUs
- CUDA toolkit installed
- Sufficient GPU memory

## Future Improvements

Potential enhancements:
1. Implement DistributedDataParallel for better scaling
2. Add gradient accumulation for larger effective batch sizes
3. Support for mixed-precision with different precisions per GPU
4. Automatic batch size tuning based on GPU memory
5. GPU memory profiling and optimization

## References

- [PyTorch DataParallel Documentation](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)
- [PyTorch Multi-GPU Best Practices](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html)
- [DataParallel vs DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
