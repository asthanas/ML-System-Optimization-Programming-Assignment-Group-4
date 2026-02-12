#!/usr/bin/env bash
# Quick activation helper
source venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Environment activated!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "PYTHONPATH: $PYTHONPATH"
echo ""
echo "Ready to run:"
echo "  • python experiments/quick_validation.py"
echo "  • pytest tests/ -v"
echo "  • python train.py --help"
echo ""
