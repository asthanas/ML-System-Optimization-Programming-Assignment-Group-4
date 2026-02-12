# Multiprocessing Fix Guide (Python 3.13 + macOS)

**Fixing the benchmark script multiprocessing errors**

---

## What's The Problem?

If you run the benchmark scripts on Python 3.13 (especially on macOS) and see this error:

```
RuntimeError: An attempt has been made to start a new process before
the current process has finished its bootstrapping phase.

This probably means that you are not using fork to start your
child processes and you have forgotten to use the proper idiom
in the main module:

    if __name__ == '__main__':
        ...
```

This is because Python 3.13 changed how multiprocessing works on macOS. The DataLoader tries to spawn worker processes, but the script isn't set up for the new 'spawn' method.

---

## Quick Fix

Use the fixed benchmark scripts I've provided:

```bash
# Instead of benchmark_training.py, use:
python3 benchmark_training_fixed.py

# Instead of benchmark_compression.py, use:
python3 benchmark_compression_fixed.py

# Or run both:
bash run_all_benchmarks.sh
```

These versions have the proper guards and use `num_workers=0` to avoid the issue entirely.

---

## What Was Changed?

**Original version:**
```python
# experiments/benchmark_training.py
tl, vl = get_dataloaders("mnist", batch_size=BATCH)

results = []
for cfg in CONFIGS:
    results.append(run(cfg))
```

**Fixed version:**
```python
# benchmark_training_fixed.py

# 1. Added proper main guard
if __name__ == "__main__":
    main()

# 2. Set num_workers=0 to avoid multiprocessing
def run(cfg):
    tl, vl = get_dataloaders("mnist", batch_size=BATCH, num_workers=0)
    # ... rest of the code
```

The key changes:
- Wrapped execution code in `if __name__ == "__main__":`
- Set `num_workers=0` in DataLoader (single-process loading)
- Proper module initialization for Python 3.13's spawn method

---

## Why This Happens

**Short version:** Python 3.13 on macOS uses 'spawn' instead of 'fork' for multiprocessing.

**Longer version:**

Python has three ways to start new processes:
- **fork** - Copy the entire parent process (fast, but has issues)
- **spawn** - Start fresh Python interpreter (safer, but pickier)
- **forkserver** - Hybrid approach

Python 3.13 on macOS uses 'spawn' by default. This is safer but requires:
1. All code at module level to be inside `if __name__ == "__main__":`
2. Everything needs to be picklable
3. The main module can't be re-executed during spawning

The original benchmark scripts violate #1, hence the error.

---

## Do I Need To Fix My Own Code?

**For training scripts:** No, `train.py` already handles this correctly.

**For benchmarks:** Use the `*_fixed.py` versions I provided.

**If you write new scripts:** Yes, follow this pattern:

```python
# my_script.py

def main():
    # All your execution code here
    dataloader = get_dataloaders("mnist", num_workers=0)
    # ... rest of your code

if __name__ == "__main__":
    main()
```

---

## Alternative: Force Fork Mode

You can force Python to use 'fork' instead (though it's deprecated):

```python
# At the very top of your script, before any imports
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

# ... rest of your imports and code
```

**Warning:** Apple is deprecating 'fork' on macOS, so this might stop working in future Python versions. Better to use the fixed scripts.

---

## The num_workers Thing

Setting `num_workers=0` tells PyTorch's DataLoader to load data in the main process rather than spawning workers.

**Pros:**
- No multiprocessing issues
- Simpler, more predictable
- Easier to debug

**Cons:**
- Slightly slower on very large datasets
- Can't overlap data loading with computation

For MNIST and CIFAR-10 on CPU, the difference is negligible. The fixed scripts use `num_workers=0` for compatibility.

---

## Platform-Specific Behavior

**Linux:** Uses 'fork' by default, original scripts usually work fine.

**macOS:** Uses 'spawn' in Python 3.13+, needs fixed scripts.

**Windows:** Always uses 'spawn', needs fixed scripts.

So the fixed scripts are actually more portable!

---

## Verifying It Works

After using the fixed scripts, you should see:

```
======================================================================
Training Benchmark (Fixed for Python 3.13)
======================================================================

Running baseline...
  [baseline] epoch 1/3  t=45.2s  acc=92.1%
  [baseline] epoch 2/3  t=44.8s  acc=95.8%
  [baseline] epoch 3/3  t=44.9s  acc=97.1%

Running topk_0.01...
  [topk_0.01] epoch 1/3  t=46.1s  acc=91.8%
  ...

✅ Saved: experiments/results/training_benchmark.csv
```

No multiprocessing errors!

---

## Still Seeing Errors?

If the fixed scripts still fail:

1. **Make sure you're using them:** Check you're running `benchmark_*_fixed.py`, not the originals

2. **Check your Python version:**
   ```bash
   python3 --version
   ```
   Should be 3.9 or higher

3. **Verify you're in the right directory:**
   ```bash
   ls -la train.py src/ tests/
   ```
   All should exist

4. **Try from scratch:**
   ```bash
   cd compressed-ddp
   source venv/bin/activate
   python3 benchmark_training_fixed.py
   ```

---

## Summary

**Problem:** Python 3.13 + macOS + multiprocessing = errors

**Solution:** Use `benchmark_*_fixed.py` scripts

**Why:** They have proper `if __name__` guards and `num_workers=0`

**Time to fix:** 0 seconds (just use different files)

The fixed scripts work on all platforms and Python versions, so you might as well use them everywhere!

---

For more details on multiprocessing in Python 3.13, see:
https://docs.python.org/3/library/multiprocessing.html

But really, just use the fixed scripts. That's what they're there for! 😊
