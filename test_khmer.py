"""
Test Khmer text rendering to verify font and text shaping support
"""
from PIL import Image, ImageDraw, ImageFont
import os

def test_khmer_rendering():
    """Test if Khmer text renders correctly"""
    
    # Find font
    font_paths = [
        "/System/Library/Fonts/Supplemental/Khmer MN.ttc",
        "/Library/Fonts/Khmer MN.ttc",
    ]
    
    font_path = None
    for path in font_paths:
        if os.path.exists(path):
            font_path = path
            print(f"✓ Found font: {path}")
            break
    
    if not font_path:
        print("✗ No Khmer font found!")
        return
    
    # Test words
    test_words = [
        "កុលបុត្រ",  # Your first word
        "សីហរដ្ឋ",   # Second word
        "សៀវភៅ",    # Third word
    ]
    
    # Check RAQM support
    has_raqm = False
    try:
        ImageFont.Layout.RAQM
        has_raqm = True
        print("✓ RAQM layout engine available")
    except AttributeError:
        print("✗ RAQM NOT available - this is the problem!")
        print("  Install with: brew install libraqm")
        print("  Then: pip install --force-reinstall pillow")
    
    # Try rendering
    font_size = 40
    font = ImageFont.truetype(font_path, font_size)
    
    for idx, word in enumerate(test_words):
        print(f"\nRendering: {word}")
        
        # Create image
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try with RAQM
        if has_raqm:
            try:
                draw.text((10, 30), word, font=font, fill='black', 
                         layout_engine=ImageFont.Layout.RAQM)
                output_file = f'test_khmer_raqm_{idx}.png'
                img.save(output_file)
                print(f"  Saved with RAQM: {output_file}")
            except Exception as e:
                print(f"  Error with RAQM: {e}")
        
        # Try without RAQM (will be broken)
        img2 = Image.new('RGB', (400, 100), color='white')
        draw2 = ImageDraw.Draw(img2)
        draw2.text((10, 30), word, font=font, fill='black')
        output_file2 = f'test_khmer_no_raqm_{idx}.png'
        img2.save(output_file2)
        print(f"  Saved without RAQM: {output_file2}")
    
    print("\n" + "="*60)
    if has_raqm:
        print("Check the 'test_khmer_raqm_*.png' files")
        print("They should show properly shaped Khmer text")
    else:
        print("RAQM is NOT installed - this is why text is broken!")
        print("\nTo fix:")
        print("1. brew install libraqm")
        print("2. pip install --force-reinstall pillow")
        print("3. Run this test again to verify")
    print("="*60)

if __name__ == '__main__':
    test_khmer_rendering()
