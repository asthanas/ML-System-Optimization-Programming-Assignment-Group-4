#!/usr/bin/env bash
# Manual MNIST downloader (bypasses SSL certificate issues)
set -euo pipefail

echo "=========================================="
echo "Manual MNIST Dataset Download"
echo "=========================================="
echo ""

# Create directory structure
mkdir -p data/MNIST/raw
cd data/MNIST/raw

echo "Downloading MNIST dataset files..."
echo ""

# Try secure URLs first, fallback to insecure if needed
download_file() {
    local url="$1"
    local filename="$2"

    echo "[$filename]"

    # Try with curl (most common on macOS)
    if command -v curl &> /dev/null; then
        # Try secure first
        if curl -L -o "$filename" "$url" 2>/dev/null; then
            echo "  ✅ Downloaded with curl"
            return 0
        fi
        # Try insecure fallback
        if curl -L -k -o "$filename" "$url" 2>/dev/null; then
            echo "  ✅ Downloaded with curl (insecure)"
            return 0
        fi
    fi

    # Try with wget
    if command -v wget &> /dev/null; then
        if wget -q -O "$filename" "$url" 2>/dev/null; then
            echo "  ✅ Downloaded with wget"
            return 0
        fi
        if wget --no-check-certificate -q -O "$filename" "$url" 2>/dev/null; then
            echo "  ✅ Downloaded with wget (insecure)"
            return 0
        fi
    fi

    echo "  ❌ Failed to download"
    return 1
}

# MNIST files
BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"

download_file "$BASE_URL/train-images-idx3-ubyte.gz" "train-images-idx3-ubyte.gz"
download_file "$BASE_URL/train-labels-idx1-ubyte.gz" "train-labels-idx1-ubyte.gz"
download_file "$BASE_URL/t10k-images-idx3-ubyte.gz" "t10k-images-idx3-ubyte.gz"
download_file "$BASE_URL/t10k-labels-idx1-ubyte.gz" "t10k-labels-idx1-ubyte.gz"

cd ../../..

echo ""
echo "=========================================="
echo "✅ MNIST Download Complete!"
echo "=========================================="
echo ""
echo "Files saved to: data/MNIST/raw/"
ls -lh data/MNIST/raw/
echo ""
echo "Now run: python3 train.py --model simple_cnn --dataset mnist --epochs 5"
echo ""
