"""
Test Khmer subscript rendering specifically
"""
from PIL import Image, ImageDraw, ImageFont
import os

def test_subscripts():
    """Test if Khmer subscripts (coeng) render correctly"""
    
    font_path = "fonts/NotoSansKhmer-Regular.ttf"
    
    if not os.path.exists(font_path):
        print("Font not found!")
        return
    
    # Test words with subscripts
    test_words = [
        ("អុបធីម៉ៃសេស្យុង", "word with ស្យ subscript"),
        ("ក្ដីស្រលាញ់", "word with ក្ដ and ស្រ subscripts"),
        ("ប្រព័ន្ធ", "word with ប្រ and ន្ធ subscripts"),
        ("ស្វ័យប្រវត្តិ", "word with ស្វ and ប្រ and ត្តិ subscripts"),
    ]
    
    font_size = 48
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Error loading font: {e}")
        return
    
    print("Testing subscript rendering...\n")
    
    for idx, (word, description) in enumerate(test_words):
        print(f"Test {idx+1}: {description}")
        print(f"  Text: {word}")
        
        # Create image with explicit language and layout engine
        img = Image.new('RGB', (600, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            # Method 1: RAQM with Khmer language specification
            draw.text((10, 30), word, font=font, fill='black',
                     layout_engine=ImageFont.Layout.RAQM,
                     language='km',
                     direction='ltr')
            output_file = f'test_subscript_{idx}_raqm_km.png'
            img.save(output_file)
            print(f"  ✓ Saved with RAQM+km: {output_file}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "="*60)
    print("Check the generated images to verify subscripts render correctly.")
    print("Subscripts should appear BELOW the base consonant, not separate.")
    print("="*60)

if __name__ == '__main__':
    test_subscripts()
