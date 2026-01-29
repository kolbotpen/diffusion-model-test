# Handwriting Generation with Diffusion Models

This project includes a diffusion model trained on the IAM Handwriting Database to generate realistic English handwriting from text prompts.

## Setup

### 1. Install Dependencies

Make sure you have the required packages:

```bash
pip install torch torchvision diffusers pillow matplotlib tqdm numpy
```

### 2. Download IAM Handwriting Database

The IAM Handwriting Database can be obtained from: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

**Steps:**

1. Register for an account on the IAM website
2. Download the following files:
   - `words.tgz` - Word images
   - `words.txt` - Metadata file

3. Extract and organize the data:

```bash
# Create directory structure
mkdir -p data/archive/iam_words

# Extract the words archive into data/archive/iam_words/
# This should create data/archive/iam_words/words/ directory with subdirectories

# Copy the metadata file
cp /path/to/words.txt data/archive/iam_words/words.txt
```

**Expected directory structure:**

```
data/archive/iam_words/
├── words.txt          # Metadata file
└── words/             # Word images
    ├── a01/
    │   ├── a01-000u/
    │   │   ├── a01-000u-00-00.png
    │   │   ├── a01-000u-00-01.png
    │   │   └── ...
    │   └── ...
    ├── a02/
    └── ...
```

## Training

Train the handwriting diffusion model:

```bash
python3 train_handwriting.py
```

**Training options:**

- Modify parameters in `train_handwriting.py` or edit the main section:
  - `num_epochs`: Number of training epochs (default: 100)
  - `batch_size`: Batch size (default: 32)
  - `learning_rate`: Learning rate (default: 1e-4)
  - `num_timesteps`: Diffusion timesteps (default: 1000)

**Note:** Training without the IAM dataset will create dummy data for testing purposes.

## Generating Handwriting

Once trained, generate handwriting from text:

```bash
# Generate the word "handwriting"
python3 evaluate_handwriting.py handwriting

# Generate any text
python3 evaluate_handwriting.py "hello world"

# Generate with custom settings
python3 evaluate_handwriting.py test --steps 500 --samples 4

# Show the denoising process in real-time
python3 evaluate_handwriting.py hello --show-process
```

**Options:**

- `text`: Text to generate (positional argument)
- `--model`: Path to model checkpoint (default: `output/handwriting/model_final.pt`)
- `--steps`: Number of denoising steps (default: 1000, lower is faster)
- `--samples`: Generate multiple variations (default: 1)
- `--output`: Custom output path for the image
- `--show-process`: Display the denoising process live
- `--device`: Force specific device (cuda, mps, cpu)

## Examples

```bash
# Quick generation (faster, lower quality)
python3 evaluate_handwriting.py "quick test" --steps 100

# High quality generation
python3 evaluate_handwriting.py "beautiful" --steps 1000

# Generate 4 variations
python3 evaluate_handwriting.py "hello" --samples 4

# Watch the generation process
python3 evaluate_handwriting.py "AI" --show-process --steps 200
```

## Output

Generated images are saved to:
- `output/handwriting/generated/<text>.png`

Training checkpoints are saved to:
- `output/handwriting/model_epoch_X.pt`
- `output/handwriting/model_final.pt`

## Model Architecture

- **Network:** UNet with text conditioning
- **Input:** Text prompts (up to 32 characters)
- **Output:** Grayscale images (64x256 pixels)
- **Conditioning:** Character-level text embeddings
- **Scheduler:** DDPM with squared cosine schedule

## Files

- `network_handwriting.py`: UNet architecture for handwriting
- `dataset_handwriting.py`: IAM dataset loader
- `train_handwriting.py`: Training script
- `evaluate_handwriting.py`: Generation/evaluation script

## Notes

- The model works best with words seen during training
- Longer words may be truncated or compressed
- Generation quality improves with more denoising steps
- GPU/MPS acceleration recommended for faster training and inference
- Without IAM data, the model will use dummy data (for testing only)

## Troubleshooting

**Model not found error:**
```bash
# Make sure you've trained the model first
python3 train_handwriting.py
```

**Out of memory:**
- Reduce `batch_size` in training
- Use fewer `--steps` during generation
- Switch to CPU with `--device cpu`

**Poor generation quality:**
- Train for more epochs
- Use more denoising steps (--steps 1000)
- Ensure IAM dataset is properly loaded
