# Data Preprocessing Pipeline

Automated pipeline for generating labeled handwriting datasets from synthetic PDFs.

## Overview

This pipeline creates labeled training data by:
1. **Generating PDFs** with random words paired with ID numbers
2. **Converting PDFs** to PNG images
3. **Extracting text boxes** from images
4. **Reading ID numbers** via OCR
5. **Creating labeled dataset** (image → word mapping)

## Pipeline Components

### 1. Word List (`data/word_list.csv`)
CSV file containing word IDs and corresponding words:
```csv
id,word
1,hello
2,world
3,machine
...
```

**You can customize this file** with your own words.

### 2. PDF Generator (`generate_pdf.py`)
Generates PDFs with format: `ID word` (one per line)

**Usage:**
```bash
# Basic usage
python generate_pdf.py

# Custom options
python generate_pdf.py \
  --word-list data/word_list.csv \
  --output data/my_pdf.pdf \
  --num-words 50 \
  --font-size 24 \
  --seed 42 \
  --convert-png
```

**Options:**
- `--word-list`: Path to word list CSV
- `--output`: Output PDF path
- `--num-words`: Number of words to include
- `--font-size`: Text font size
- `--seed`: Random seed for reproducibility
- `--convert-png`: Also convert to PNG

**Outputs:**
- PDF file with labeled words
- Metadata CSV with word ordering
- PNG image (if `--convert-png` used)

### 3. Extract and Label (`extract_and_label.py`)
Extracts text boxes, reads IDs via OCR, and creates labeled dataset.

**Usage:**
```bash
# Basic usage
python extract_and_label.py generated.png

# Custom options
python extract_and_label.py generated.png \
  --word-list data/word_list.csv \
  --output labeled_data \
  --min-area 1000 \
  --min-aspect-ratio 2.0 \
  --debug
```

**Options:**
- `image`: Input PNG image
- `--word-list`: Path to word list CSV
- `--output`: Output directory
- `--min-area`: Minimum box area threshold
- `--min-aspect-ratio`: Minimum width/height ratio
- `--debug`: Save intermediate OCR images

**Outputs:**
- `labels.csv`: Labeled dataset (image_path, word_id, word_text)
- `images/`: Directory with extracted word images
- `visualization.png`: Visualization of detected boxes

### 4. Complete Pipeline (`pipeline.py`)
Orchestrates the entire process end-to-end.

**Usage:**
```bash
# Basic usage (generates 30 labeled samples)
python pipeline.py

# Custom configuration
python pipeline.py \
  --word-list data/word_list.csv \
  --num-words 100 \
  --output data/pipeline_output \
  --seed 42 \
  --debug
```

**Options:**
- `--word-list`: Path to word list CSV
- `--num-words`: Number of words to generate
- `--output`: Output base directory
- `--seed`: Random seed
- `--min-area`: Minimum box area
- `--min-aspect-ratio`: Minimum aspect ratio
- `--debug`: Enable debug mode

**Outputs:**
- `generated.pdf`: Generated PDF
- `generated.png`: PNG conversion
- `labeled_data/labels.csv`: Labeled dataset
- `labeled_data/images/`: Extracted word images
- `labeled_data/visualization.png`: Box detection visualization

## Installation

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install system dependencies

**macOS:**
```bash
brew install poppler tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils tesseract-ocr
```

**Windows:**
- Install Poppler: https://github.com/oschwartz10612/poppler-windows
- Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki

## Quick Start

### Step 1: Prepare Word List
Edit `data/word_list.csv` with your words:
```csv
id,word
1,apple
2,banana
3,cherry
```

### Step 2: Run Pipeline
```bash
python pipeline.py --num-words 30 --seed 42
```

### Step 3: Use Labeled Data
The labeled data is saved in `data/pipeline_output/labeled_data/`:
- `labels.csv`: CSV with image paths and labels
- `images/`: Directory with word images

**Example `labels.csv`:**
```csv
image_path,word_id,word_text
images/word_0015_000.png,15,apple
images/word_0023_001.png,23,banana
...
```

## Advanced Usage

### Generate Multiple Datasets
```bash
# Dataset 1
python pipeline.py --output data/dataset1 --seed 1 --num-words 50

# Dataset 2
python pipeline.py --output data/dataset2 --seed 2 --num-words 50

# Dataset 3
python pipeline.py --output data/dataset3 --seed 3 --num-words 50
```

### Custom Extraction Parameters
If boxes aren't being detected correctly, adjust:
```bash
python pipeline.py \
  --min-area 500 \        # Lower for smaller text
  --min-aspect-ratio 1.5  # Lower for more square boxes
```

### Debug Mode
To see intermediate OCR images:
```bash
python pipeline.py --debug
```

This saves:
- `debug_id_box.png`: Raw ID region extraction
- `debug_id_processed.png`: Preprocessed for OCR

## Troubleshooting

### PDF to PNG conversion fails
**Error:** `pdf2image not installed` or `poppler not found`

**Solution:**
```bash
pip install pdf2image
brew install poppler  # macOS
```

### OCR not reading IDs correctly
**Problem:** IDs are not extracted or wrong numbers

**Solutions:**
1. Check Tesseract installation: `tesseract --version`
2. Use `--debug` to see OCR preprocessing
3. Adjust font size in PDF generation: `--font-size 32`
4. Ensure sufficient spacing between ID and word

### No boxes detected
**Problem:** `Found 0 text boxes`

**Solutions:**
1. Lower `--min-area` threshold (try 500)
2. Lower `--min-aspect-ratio` (try 1.5)
3. Check PNG image quality
4. Ensure PDF has visible text (not image-based)

### Boxes overlap or duplicate
**Problem:** Multiple boxes for same word

**Solution:**
The pipeline filters overlapping boxes automatically. If issues persist, boxes may be too close together in the PDF.

## File Structure

```
diffusion-model-test/
├── data/
│   ├── word_list.csv              # Your word list
│   └── pipeline_output/           # Pipeline outputs
│       ├── generated.pdf
│       ├── generated.png
│       ├── generated.csv          # Metadata
│       └── labeled_data/
│           ├── labels.csv         # Labeled dataset
│           ├── images/            # Word images
│           └── visualization.png
├── generate_pdf.py                # PDF generator
├── extract_and_label.py           # Box extraction & labeling
├── pipeline.py                    # Main pipeline orchestrator
└── PIPELINE_README.md             # This file
```

## Integration with Training

To use this labeled data for training your model:

```python
import pandas as pd
from PIL import Image

# Load labels
df = pd.read_csv('data/pipeline_output/labeled_data/labels.csv')

for idx, row in df.iterrows():
    image_path = row['image_path']
    word_id = row['word_id']
    word_text = row['word_text']
    
    # Load image (relative to labeled_data directory)
    full_path = f"data/pipeline_output/labeled_data/{image_path}"
    image = Image.open(full_path)
    
    # Your training code here
    # train(image, label=word_text)
```

## Tips for Best Results

1. **Font Size**: Use larger fonts (24-32pt) for better OCR accuracy
2. **Spacing**: Ensure adequate spacing between ID and word in PDF
3. **Contrast**: High contrast helps both OCR and box detection
4. **Reproducibility**: Use `--seed` for consistent results
5. **Batch Processing**: Generate multiple datasets with different seeds

## Next Steps

After generating labeled data:

1. **Verify labels**: Check `visualization.png` and sample images
2. **Combine datasets**: Merge multiple pipeline runs for larger datasets
3. **Train models**: Use labeled data for handwriting recognition
4. **Iterate**: Adjust parameters based on results

## Support

For issues or questions:
1. Check visualization to see if boxes are detected correctly
2. Use `--debug` mode to inspect OCR processing
3. Verify word_list.csv format is correct
4. Ensure all dependencies are installed
