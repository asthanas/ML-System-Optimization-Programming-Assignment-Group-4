# SSL Certificate Fix (macOS)

**Quick reference for the SSL certificate issue**

---

## What's Happening?

If you're on macOS with Python 3.13 and you see this error:

```
RuntimeError: Error downloading train-images-idx3-ubyte.gz:
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed
```

Don't worry - it's a known issue with how Python 3.13 handles SSL certificates on macOS. The MNIST dataset download is failing because Python can't verify the SSL certificate.

---

## Quick Fix (30 seconds)

The fastest solution is to download MNIST manually:

```bash
bash download_mnist.sh
python3 train.py --model simple_cnn --dataset mnist --epochs 5
```

That's it! The script downloads the files with SSL verification disabled, then training works normally.

---

## What download_mnist.sh Does

It creates the `data/MNIST/raw/` directory and downloads these 4 files:
- `train-images-idx3-ubyte.gz` (9.9 MB)
- `train-labels-idx1-ubyte.gz` (28 KB)
- `t10k-images-idx3-ubyte.gz` (1.6 MB)
- `t10k-labels-idx1-ubyte.gz` (4.5 KB)

Total: about 12 MB. Takes ~30 seconds.

Once downloaded, PyTorch uses the local files and you won't see the error again.

---

## Alternative Solutions

### Python Version (if bash doesn't work)

```bash
python3 download_mnist.py
python3 train.py --model simple_cnn --dataset mnist --epochs 5
```

Same thing, just pure Python instead of bash.

### Permanent Fix

Install Python's SSL certificates properly:

```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

This runs Python's certificate installer. After this, downloads should work normally without any special scripts.

### Manual Download (if scripts fail)

Download files directly with curl:

```bash
mkdir -p data/MNIST/raw
cd data/MNIST/raw

curl -L -k -o train-images-idx3-ubyte.gz \
  https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz

curl -L -k -o train-labels-idx1-ubyte.gz \
  https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz

curl -L -k -o t10k-images-idx3-ubyte.gz \
  https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz

curl -L -k -o t10k-labels-idx1-ubyte.gz \
  https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz

cd ../../..
```

The `-k` flag tells curl to skip SSL verification.

---

## Verify It Worked

Check that the files are there:

```bash
ls -lh data/MNIST/raw/
```

You should see all 4 `.gz` files. If so, training will work!

---

## Why This Happens

Python 3.13 on macOS doesn't ship with SSL certificates configured by default. This is a Python packaging issue, not a problem with your system or the code.

The fix (downloading manually or installing certificates) is a one-time thing. Once done, you won't see this error again.

---

## Still Having Issues?

If none of these work, check QUICK_START_GUIDE.md Section 5 for more troubleshooting. But honestly, the bash script should do it.

Happy training!
