# Quick Reference Card - Compressed-DDP

## Setup (One Time)
```bash
cd compressed-ddp
bash setup.sh
```

## Activate (Every Session)
```bash
source activate.sh
```

## Verify Installation
```bash
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
```

## Train Models

### MNIST (Quick Test - 3 min)
```bash
python train.py --model simple_cnn --dataset mnist --epochs 5
```

### CIFAR-10 (Full Test - 15 min)
```bash
python train.py --model simple_cnn --dataset cifar10 --epochs 10
```

### Custom Training
```bash
python train.py --model MODEL --dataset DATASET --epochs N --batch-size B --lr LR
```

## Run Tests
```bash
pytest tests/ -v                    # All tests
pytest tests/test_compression.py    # Specific file
pytest tests/ --cov=src             # With coverage
```

## Experiments
```bash
python experiments/quick_validation.py
```

## View Logs
```bash
tensorboard --logdir=runs
# Open: http://localhost:6006
```

## Troubleshooting

### Import Error
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

### Clean Cache
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
```

### Reset Environment
```bash
rm -rf venv && bash setup.sh
```

## File Locations

- Models: `checkpoints/`
- Datasets: `data/`
- Logs: `runs/`
- Source: `src/`

## Expected Results

✅ Setup: 2-3 minutes  
✅ Training (MNIST, 5 epochs): ~3 minutes  
✅ Tests: ~5 seconds  
✅ Quick validation: ~30 seconds  

## Support

See COMMAND_GUIDE.md for detailed explanations.
