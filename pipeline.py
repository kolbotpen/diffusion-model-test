"""
Complete Data Preprocessing Pipeline
Orchestrates PDF generation, conversion, extraction, and labeling
"""
import os
import argparse
from pathlib import Path
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n✗ Error in: {description}")
        sys.exit(1)
    
    print(f"\n✓ Completed: {description}")
    return result


def run_pipeline(word_list, num_words, output_base, seed=None, 
                 min_area=1000, min_aspect_ratio=2.0, debug=False):
    """
    Run the complete data preprocessing pipeline.
    
    Pipeline steps:
    1. Generate PDF with random words and IDs
    2. Convert PDF to PNG
    3. Extract boxes and label them
    
    Args:
        word_list: Path to word list CSV
        num_words: Number of words to include
        output_base: Base directory for all outputs
        seed: Random seed for reproducibility
        min_area: Minimum box area for extraction
        min_aspect_ratio: Minimum aspect ratio for extraction
        debug: Enable debug mode
    """
    # Setup paths
    output_path = Path(output_base)
    output_path.mkdir(exist_ok=True, parents=True)
    
    pdf_path = output_path / 'generated.pdf'
    png_path = output_path / 'generated.png'
    labeled_dir = output_path / 'labeled_data'
    
    # Step 1: Generate PDF
    cmd = [
        sys.executable, 'generate_pdf.py',
        '--word-list', word_list,
        '--output', str(pdf_path),
        '--num-words', str(num_words),
        '--convert-png',
    ]
    if seed is not None:
        cmd.extend(['--seed', str(seed)])
    
    run_command(cmd, "Generate PDF with labeled words")
    
    # Verify PNG was created
    if not png_path.exists():
        print(f"\n✗ Error: PNG not created at {png_path}")
        print("Make sure pdf2image and poppler are installed:")
        print("  pip install pdf2image")
        print("  brew install poppler  (macOS)")
        sys.exit(1)
    
    # Step 2: Extract and label boxes
    cmd = [
        sys.executable, 'extract_and_label.py',
        str(png_path),
        '--word-list', word_list,
        '--output', str(labeled_dir),
        '--min-area', str(min_area),
        '--min-aspect-ratio', str(min_aspect_ratio),
    ]
    if debug:
        cmd.append('--debug')
    
    run_command(cmd, "Extract and label text boxes")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Pipeline completed successfully!")
    print(f"{'='*60}")
    print(f"\nGenerated files:")
    print(f"  PDF:              {pdf_path}")
    print(f"  PNG:              {png_path}")
    print(f"  Labels CSV:       {labeled_dir / 'labels.csv'}")
    print(f"  Labeled images:   {labeled_dir / 'images/'}")
    print(f"  Visualization:    {labeled_dir / 'visualization.png'}")
    
    # Count labeled images
    images_dir = labeled_dir / 'images'
    if images_dir.exists():
        num_images = len(list(images_dir.glob('*.png')))
        print(f"\nTotal labeled images: {num_images}")
    
    return labeled_dir


def main():
    parser = argparse.ArgumentParser(
        description='Complete data preprocessing pipeline for handwriting dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Generate 30 words with default settings
  python pipeline.py
  
  # Generate 50 words with custom word list
  python pipeline.py --num-words 50 --word-list my_words.csv
  
  # Use seed for reproducibility
  python pipeline.py --seed 42
  
  # Debug mode (saves intermediate OCR images)
  python pipeline.py --debug
        """
    )
    
    parser.add_argument('--word-list', '-w',
                        default='data/word_list.csv',
                        help='Path to word list CSV (default: data/word_list.csv)')
    parser.add_argument('--num-words', '-n', type=int, default=10,
                        help='Number of words to generate (default: 10)')
    parser.add_argument('--output', '-o',
                        default='data/pipeline_output',
                        help='Output directory (default: data/pipeline_output)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--min-area', type=int, default=1000,
                        help='Minimum box area for extraction (default: 1000)')
    parser.add_argument('--min-aspect-ratio', type=float, default=2.0,
                        help='Minimum aspect ratio for extraction (default: 2.0)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Verify word list exists
    if not Path(args.word_list).exists():
        print(f"Error: Word list not found: {args.word_list}")
        print("\nYou need a CSV file with columns: id,word")
        print("Example:")
        print("  id,word")
        print("  1,hello")
        print("  2,world")
        sys.exit(1)
    
    try:
        run_pipeline(
            word_list=args.word_list,
            num_words=args.num_words,
            output_base=args.output,
            seed=args.seed,
            min_area=args.min_area,
            min_aspect_ratio=args.min_aspect_ratio,
            debug=args.debug
        )
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
