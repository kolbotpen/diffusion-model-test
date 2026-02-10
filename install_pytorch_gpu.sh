#!/bin/bash
# Bash script to install PyTorch with GPU support
# For Mac (Apple Silicon with MPS)

echo ""
echo "============================================================"
echo "PyTorch GPU Installation"
echo "============================================================"
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠ Warning: No virtual environment detected"
    echo "  Activating venv..."
    source venv/bin/activate
else
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
fi

echo ""
echo "Step 1: Uninstalling existing PyTorch..."
pip uninstall -y torch torchvision torchaudio

echo ""
echo "Step 2: Installing PyTorch with MPS support (Apple Silicon)..."
echo "(For Mac with Apple Silicon M1/M2/M3)"
echo ""

# Install PyTorch with MPS support (default for Apple Silicon)
pip install torch torchvision torchaudio

echo ""
echo "============================================================"
echo "Installation complete! Verifying..."
echo "============================================================"
echo ""

# Verify installation
python test_gpu.py

echo ""
echo "============================================================"
echo "Next steps:"
echo "  If MPS is available, run: python train_handwriting.py"
echo "  If MPS is NOT available, check:"
echo "    1. You're using macOS 12.3 or later"
echo "    2. You have an Apple Silicon Mac (M1/M2/M3)"
echo "============================================================"
echo ""
