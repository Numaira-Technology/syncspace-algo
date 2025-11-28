"""
Quick test script to verify vision-based processing setup.

Run this to check if all dependencies are installed and configured correctly.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        print("✓ Pillow (PIL) installed")
    except ImportError as e:
        print(f"✗ Pillow not installed: {e}")
        return False
    
    try:
        from openpyxl import load_workbook
        print("✓ openpyxl installed")
    except ImportError as e:
        print(f"✗ openpyxl not installed: {e}")
        return False
    
    try:
        from docx import Document
        print("✓ python-docx installed")
    except ImportError as e:
        print(f"✗ python-docx not installed: {e}")
        return False
    
    try:
        from openai import OpenAI
        print("✓ openai installed")
    except ImportError as e:
        print(f"✗ openai not installed: {e}")
        return False
    
    try:
        import httpx
        print("✓ httpx installed")
    except ImportError as e:
        print(f"✗ httpx not installed: {e}")
        return False
    
    return True

def test_environment():
    """Test environment variables."""
    print("\nTesting environment variables...")
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key:
        print(f"✓ OPENROUTER_API_KEY is set ({api_key[:10]}...)")
        return True
    else:
        print("✗ OPENROUTER_API_KEY not set")
        print("  Set it with: export OPENROUTER_API_KEY=sk-xxx")
        return False

def test_modules():
    """Test that custom modules can be imported."""
    print("\nTesting custom modules...")
    
    try:
        from utils.excel_to_image import convert_excel_to_image
        print("✓ excel_to_image module")
    except ImportError as e:
        print(f"✗ excel_to_image module: {e}")
        return False
    
    try:
        from utils.vision_data_extractor import extract_data_from_excel_image
        print("✓ vision_data_extractor module")
    except ImportError as e:
        print(f"✗ vision_data_extractor module: {e}")
        return False
    
    try:
        from utils.document_processing.vision_processor import process_files_with_vision
        print("✓ vision_processor module")
    except ImportError as e:
        print(f"✗ vision_processor module: {e}")
        return False
    
    try:
        from funnels.llm_provider import LLMProvider
        print("✓ llm_provider module")
    except ImportError as e:
        print(f"✗ llm_provider module: {e}")
        return False
    
    return True

def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    
    try:
        from config import Config
        config = Config()
        llm_config = config.get_model_config()["llm"]
        
        print(f"✓ Config loaded")
        print(f"  Provider: {llm_config['provider']}")
        print(f"  Model: {llm_config['model']}")
        print(f"  Timeout: {llm_config['timeout']}s")
        
        if llm_config['model'] == "google/gemini-2.0-flash-exp:free":
            print("✓ Using correct vision model (Gemini 2.0 Flash)")
        else:
            print(f"⚠ Using model: {llm_config['model']}")
            print("  Recommended: google/gemini-2.0-flash-exp:free")
        
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def test_image_creation():
    """Test basic image creation."""
    print("\nTesting image creation...")
    
    try:
        from PIL import Image, ImageDraw
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 90, 90], outline='black')
        
        print("✓ Can create images with Pillow")
        return True
    except Exception as e:
        print(f"✗ Image creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("SYNCSPACE ALGO - VISION SETUP TEST")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Environment", test_environment()))
    results.append(("Modules", test_modules()))
    results.append(("Config", test_config()))
    results.append(("Image Creation", test_image_creation()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - System is ready!")
        print("\nYou can now run:")
        print("  python main.py <docx_file> <excel_file>")
        print("  python app.py  # Web interface")
        print("  python api.py  # REST API")
    else:
        print("✗ SOME TESTS FAILED - Please fix the issues above")
        print("\nCommon fixes:")
        print("  pip install -r requirements.txt")
        print("  export OPENROUTER_API_KEY=sk-xxx")
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

