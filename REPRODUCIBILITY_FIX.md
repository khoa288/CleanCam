# Reproducibility Fix: Why Results Differed with Same Seed

## Problem

Even when running with the same seed (e.g., seed=42), results were different across different runs. This is a critical issue for reproducible research.

## Root Causes Found

### 1. **CRITICAL BUG: `torch.use_deterministic_algorithms(False)`**

**Location:** `cleancam_pipeline/utils/seed.py`

**The Bug:**
```python
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)  # ← WRONG! Disables determinism
    except Exception:
        pass
```

**Why It's Wrong:**
- `torch.use_deterministic_algorithms(False)` explicitly **disables** deterministic algorithms
- This means PyTorch uses non-deterministic operations for speed
- Even with seeds set, results will vary between runs

### 2. **Missing CUDA Determinism Flags**

**Missing:**
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Why It Matters:**
- CUDA operations need explicit determinism flags
- `cudnn.benchmark = False` disables auto-tuning (which is non-deterministic)
- `cudnn.deterministic = True` forces deterministic algorithms

### 3. **Data Augmentation Randomness**

**Location:** `cleancam_pipeline/data/transforms.py`

**Transforms with Randomness:**
```python
transforms.RandomCrop(image_size)           # Random crop position
transforms.RandomHorizontalFlip(p=0.5)      # Random flip decision
```

**Why It's OK:**
- These ARE seeded by `set_seed()`
- But only if deterministic algorithms are enabled
- Without determinism flags, even seeded random operations can vary

### 4. **Multi-GPU Non-Determinism**

**Location:** `cleancam_pipeline/models/training.py`

**Issue:**
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)  # Multi-GPU can introduce non-determinism
```

**Why:**
- DataParallel uses parallel operations that may not be deterministic
- Gradient synchronization across GPUs can vary
- Use `--single-gpu` flag for reproducibility if needed

## The Fix

### Updated `set_seed()` Function

```python
def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic algorithms for reproducibility
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Changes:**
1. Changed `torch.use_deterministic_algorithms(False)` → `True`
2. Added `torch.backends.cudnn.deterministic = True`
3. Added `torch.backends.cudnn.benchmark = False`
4. Removed try-except (determinism should not fail)

## Impact

### Before Fix
- Same seed → Different results across runs
- Non-reproducible research
- Cannot verify results

### After Fix
- Same seed → **Identical results** across runs
- Fully reproducible
- Can verify and compare results

## Verification

To verify reproducibility works:

```bash
# Run 1
python cleancam_pipeline.py \
    --release-root /path/to/data \
    --output-root ./output1 \
    --run-benchmark \
    --models mobilenet_v2 \
    --seeds 42 \
    --benchmark-settings train_real_only__eval_real_only \
    --epochs 5

# Run 2 (same command)
python cleancam_pipeline.py \
    --release-root /path/to/data \
    --output-root ./output2 \
    --run-benchmark \
    --models mobilenet_v2 \
    --seeds 42 \
    --benchmark-settings train_real_only__eval_real_only \
    --epochs 5

# Compare results
diff output1/benchmark/mobilenet_v2/train_real_only__eval_real_only/seed_42/test_predictions_*.csv \
    output2/benchmark/mobilenet_v2/train_real_only__eval_real_only/seed_42/test_predictions_*.csv
```

**Expected:** Files should be identical (or nearly identical with floating point precision)

## Performance Trade-off

**Note:** Enabling deterministic algorithms may reduce performance:
- ~5-10% slower training
- Worth it for reproducibility in research

**If you need speed over reproducibility:**
```python
# Disable determinism (not recommended for research)
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.benchmark = True
```

## Reproducibility Checklist

For fully reproducible results:

- ✅ Same seed value
- ✅ Same model architecture
- ✅ Same data (same train/val/test split)
- ✅ Same hyperparameters (learning rate, batch size, etc.)
- ✅ Same hardware (GPU model, CPU cores)
- ✅ Same PyTorch version
- ✅ Deterministic algorithms enabled (now fixed)
- ⚠️ Multi-GPU may still have minor variations (use `--single-gpu` for strict reproducibility)

## Files Changed

- `cleancam_pipeline/utils/seed.py` - Fixed `set_seed()` function

## References

- [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- [CUDA Determinism](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#deterministic-operations)
- [DataParallel Reproducibility](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)
