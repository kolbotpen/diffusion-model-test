# PowerShell script to install PyTorch with CUDA support for RTX 4060
# This will uninstall the CPU-only version and install the CUDA version

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "PyTorch CUDA Installation for RTX 4060" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

# Check if virtual environment is activated
if ($env:VIRTUAL_ENV) {
    Write-Host "✓ Virtual environment detected: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "⚠ Warning: No virtual environment detected" -ForegroundColor Yellow
    Write-Host "  Activating venv..." -ForegroundColor Yellow
    & ".\venv\Scripts\Activate.ps1"
}

Write-Host "`nStep 1: Uninstalling CPU-only PyTorch..." -ForegroundColor Yellow
pip uninstall -y torch torchvision torchaudio

Write-Host "`nStep 2: Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Yellow
Write-Host "(This version is compatible with RTX 4060)`n" -ForegroundColor Gray

# Install PyTorch with CUDA 12.1 (compatible with RTX 4060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Installation complete! Verifying..." -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

# Verify installation
python test_gpu.py

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  If CUDA is available, run: python train_handwriting.py" -ForegroundColor White
Write-Host "  If CUDA is NOT available, check:" -ForegroundColor White
Write-Host "    1. NVIDIA drivers are installed and updated" -ForegroundColor White
Write-Host "    2. CUDA toolkit is installed (optional)" -ForegroundColor White
Write-Host "    3. Your GPU appears in Device Manager" -ForegroundColor White
Write-Host "============================================================`n" -ForegroundColor Cyan
