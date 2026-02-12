#!/usr/bin/env bash
# Patch script to fix MPS pin_memory warning

echo "════════════════════════════════════════════════════════════"
echo "Patching src/data/loaders.py to fix MPS pin_memory warning"
echo "════════════════════════════════════════════════════════════"
echo ""

# Backup original
if [ -f "src/data/loaders.py" ]; then
    echo "📋 Backing up original file..."
    cp src/data/loaders.py src/data/loaders.py.backup
    echo "   ✅ Backup: src/data/loaders.py.backup"
fi

# Apply patch
if [ -f "loaders_fixed.py" ]; then
    echo ""
    echo "📝 Applying patch..."
    cp loaders_fixed.py src/data/loaders.py
    echo "   ✅ Patched: src/data/loaders.py"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "✅ Patch applied successfully!"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "The pin_memory warning will no longer appear."
    echo ""
    echo "Test it:"
    echo "  python3 train.py --model simple_cnn --dataset mnist --epochs 5"
    echo ""
else
    echo "❌ Error: loaders_fixed.py not found"
    exit 1
fi
