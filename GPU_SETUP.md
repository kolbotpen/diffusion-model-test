# GPU Setup Instructions

Your code has been **fixed** to work properly with both Windows (CUDA) and Mac (MPS) GPUs. However, you currently have PyTorch installed **without GPU support**.

## Current Issue

You have PyTorch version `2.8.0+cpu` which doesn't include CUDA support for your RTX 4060.

## Solution

### For Windows (NVIDIA GPU like your RTX 4060)

#### Quick Install (Recommended)
Run the provided PowerShell script:
```powershell
.\install_pytorch_gpu.ps1
```

#### Manual Install
If you prefer to install manually:
```powershell
# Activate your virtual environment
.\venv\Scripts\Activate.ps1

# Uninstall CPU-only PyTorch
pip uninstall -y torch torchvision torchaudio

# Install PyTorch with CUDA 12.1 (compatible with RTX 4060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python test_gpu.py
```

### For Mac (Apple Silicon M1/M2/M3)

#### Quick Install (Recommended)
Run the provided bash script:
```bash
chmod +x install_pytorch_gpu.sh
./install_pytorch_gpu.sh
```

#### Manual Install
```bash
# Activate your virtual environment
source venv/bin/activate

# Uninstall existing PyTorch
pip uninstall -y torch torchvision torchaudio

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Verify installation
python test_gpu.py
```

## Verification

After installation, run:
```bash
python test_gpu.py
```

You should see:
- **Windows**: `CUDA available: True` and your GPU name (RTX 4060)
- **Mac**: `MPS (Apple Metal) available: True`

## Training

Once GPU support is confirmed, start training:
```bash
python train_handwriting.py
```

## Code Fixes Applied

The following bugs have been fixed:
1. ✅ **Positional encoding batching** - Fixed dimension mismatches in attention mechanism
2. ✅ **Device handling** - Improved cross-platform GPU detection
3. ✅ **MPS compatibility** - Added error handling for Apple Silicon GPUs
4. ✅ **Attention layer crash** - Fixed the `bmm` error you were experiencing

## Troubleshooting

### Windows - CUDA Not Available After Install
1. **Update NVIDIA drivers**: Visit [NVIDIA Driver Downloads](https://www.nvidia.com/download/index.aspx)
2. **Check GPU in Device Manager**: Ensure RTX 4060 appears without errors
3. **Try different CUDA version**: Some GPUs work better with CUDA 11.8
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Mac - MPS Not Available
1. **Update macOS**: MPS requires macOS 12.3 or later
2. **Verify Apple Silicon**: MPS only works on M1/M2/M3 chips
3. **Intel Mac**: Use CPU (MPS not supported on Intel Macs)

## Performance Comparison

| Device | Expected Speed | Your Setup |
|--------|---------------|------------|
| RTX 4060 (CUDA) | ~15-20x faster | ⚠ Currently CPU |
| Apple M1/M2 (MPS) | ~8-12x faster | N/A |
| CPU | 1x (baseline) | ✓ Currently active |

**Bottom line**: You're missing out on ~15-20x speedup! Install CUDA support to utilize your RTX 4060.
