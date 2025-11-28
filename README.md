# SyncSpace Algo - Vision-Based Financial Document Processor

## Overview
This system automatically updates financial documents using **vision AI** to understand Excel tables and intelligently update Word documents. Unlike traditional text-matching approaches, this system uses computer vision and large language models to comprehend the semantic meaning of financial data and make contextually appropriate updates.

## Key Features
- 🔍 **Vision-Based Understanding**: Converts Excel to images and uses vision AI to extract structured data
- 🧠 **Semantic Analysis**: AI comprehends what data means, not just pattern matching
- 📊 **Intelligent Updates**: Determines what needs updating based on context and meaning
- ✨ **Format Preservation**: Maintains original document structure and formatting
- 🎯 **High Accuracy**: Confidence scoring for each update
- 🌍 **Automatic Geo-Fallback**: Automatically switches to alternative models if primary model is geo-restricted

## Architecture

### Core Components

1. **Vision Processing Pipeline**
   - Excel to Image Converter (`utils/excel_to_image.py`): Renders Excel sheets as high-resolution images
   - Vision Data Extractor (`utils/vision_data_extractor.py`): Uses vision AI to extract structured data from images
   - Vision Processor (`utils/document_processing/vision_processor.py`): Orchestrates analysis and updates

2. **LLM Integration with Vision Support**
   - OpenRouter Integration with vision models:
     - **Google Gemini 3 Pro Preview** (Primary - best quality, vision-enabled)
     - **Qwen3-VL-8B-Instruct** (Fallback - no geo-restrictions, direct answers)
     - Supports 100+ models via unified API
   - Automatic fallback for geo-restricted regions
   - Vision API support for image analysis
   - Configurable timeout and model settings
   - Structured data extraction from images

3. **Document Processing**
   - Word Parser (`funnels/extract.py`): Extracts sentences from Word documents
   - Update Engine: Applies changes with highlighting

4. **Web Interface**
   - Flask-based web application
   - File upload/download functionality
   - Real-time processing status
   - Results visualization with confidence scores

## How It Works

### Vision-Based Pipeline

1. **Excel to Image Conversion**
   - Converts Excel spreadsheet to high-resolution PNG image
   - Preserves table structure, headers, and formatting
   - Optimized for vision model comprehension

2. **Vision-Based Data Extraction**
   - Vision AI analyzes the Excel image
   - Identifies column headers (time periods like "Q1 2023")
   - Identifies row headers (metrics like "Revenue", "Net Income")
   - Extracts all numerical values with correct associations
   - Outputs structured JSON data:
   ```json
   {
     "metrics": {
       "Revenue": {
         "Three Months Ended June 30, 2023": 26930000000,
         "Six Months Ended June 30, 2023": 42260000000
       }
     },
     "time_periods": ["Three Months Ended June 30, 2023", ...],
     "metadata": {"currency": "USD", "description": "..."}
   }
   ```

3. **Semantic Document Analysis**
   - AI reads each sentence in the Word document
   - Determines if sentence contains financial data
   - Identifies which metrics and time periods are mentioned
   - Decides what needs updating based on semantic understanding

4. **Intelligent Updates**
   - AI generates updated sentences with correct values
   - Maintains original sentence structure and tone
   - Preserves number formatting (e.g., "$X.XX billion")
   - Only updates what's necessary

5. **Confidence Scoring**
   - Length similarity: Updated sentence should be similar length
   - Metric presence: Mentioned metrics still present in update
   - Structural similarity: Overall sentence structure preserved
   - Typical confidence: 0.7-0.95

## Advantages Over Traditional Approach

| Traditional (Text Matching) | Vision-Based (This System) |
|----------------------------|----------------------------|
| Requires exact string matches | Understands semantic meaning |
| Brittle to formatting changes | Robust to different formats |
| Can't handle complex tables | Handles any table structure |
| Manual metric name mapping | Automatic comprehension |
| Limited context awareness | Full contextual understanding |

## Installation

### Prerequisites
- Python 3.8+
- OpenRouter API key (get one at https://openrouter.ai/)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd syncspace-algo

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file or export:
export OPENROUTER_API_KEY=sk-xxx  # Your OpenRouter API key
```

## Usage

### Command Line
```bash
python main.py <docx_file> <excel_file>
```

Example:
```bash
python main.py financial_report.docx quarterly_data.xlsx
```

Output:
- Updated document saved as `financial_report_updated.docx`
- Changes highlighted in yellow
- Console shows all updates with confidence scores

### Web Interface
```bash
python app.py  # Starts server on http://localhost:8080
```

Features:
- Upload Word and Excel files via web form
- View all proposed changes with confidence scores
- Download updated document with highlighted changes

### REST API (FastAPI)
```bash
python api.py  # Starts server on http://localhost:8000
```

API Documentation available at: http://localhost:8000/docs

#### Endpoints

**Process Documents**
```http
POST /run-syncspace/
Content-Type: multipart/form-data

Files:
  - docx_file: Word document (.docx)
  - excel_file: Excel spreadsheet (.xlsx)

Response:
{
  "status": "success",
  "data": {
    "number_of_changes": 5,
    "results": [
      {
        "original_text": "...",
        "modified_text": "...",
        "confidence": 0.85,
        "description": "Updated Revenue for Q1 2023"
      }
    ],
    "output_file_path": "path/to/updated.docx"
  }
}
```

## Configuration

### Model Selection

The system uses **automatic fallback** for geo-restricted regions:

```python
"llm": {
    "provider": "openrouter",
    "model": "google/gemini-3-pro-preview",  # Primary: Best quality
    "fallback_model": "qwen/qwen3-vl-8b-instruct",  # Fallback: No geo-restrictions
    "timeout": 60
}
```

**How it works:**
- System first tries the primary model (Gemini 3)
- If geo-restricted, automatically switches to fallback model (Qwen3-VL-Instruct)
- No manual intervention required

**Alternative configurations:**

For geo-restricted regions (Hong Kong, China, etc.):
```python
"model": "qwen/qwen3-vl-8b-instruct",  # Use Qwen as primary
```

For maximum reliability:
```python
"model": "anthropic/claude-3.5-sonnet:beta",  # Claude (no geo-restrictions)
```

See `AUTOMATIC_FALLBACK_GUIDE.md` for detailed information.

### Confidence Threshold
Adjust minimum confidence for updates in your code:

```python
apply_updates_to_document(
    docx_path, 
    output_path, 
    results, 
    min_confidence=0.7  # Only apply updates with 70%+ confidence
)
```

## Project Structure

```
syncspace-algo/
├── main.py                          # Command-line entry point
├── app.py                           # Flask web interface
├── api.py                           # FastAPI REST API
├── requirements.txt                 # Python dependencies
├── config/
│   └── base_config.py              # Configuration settings
├── utils/
│   ├── excel_to_image.py           # Excel → Image converter
│   ├── vision_data_extractor.py    # Vision AI data extraction
│   ├── document_processing/
│   │   └── vision_processor.py     # Main vision processing logic
│   └── logging.py                  # Logging configuration
├── funnels/
│   ├── llm_provider.py             # OpenRouter integration
│   └── extract.py                  # Word document parser
└── templates/
    └── index.html                  # Web interface template
```

## Key Components

### Vision Processing
- **`utils/excel_to_image.py`**: Converts Excel sheets to high-resolution PNG images
- **`utils/vision_data_extractor.py`**: Uses vision AI to extract structured financial data
- **`utils/document_processing/vision_processor.py`**: Main processing pipeline

### LLM Integration
- **`funnels/llm_provider.py`**: OpenRouter API integration with vision support
  - Text analysis
  - Image analysis (vision models)
  - Structured data extraction

### Document Handling
- **`funnels/extract.py`**: Word document text extraction
- **`python-docx`**: Document manipulation and highlighting

## Error Handling
- **File size limit**: 16MB per file
- **Supported formats**: .docx and .xlsx only
- **Automatic cleanup**: Temporary files removed after processing
- **Timeout handling**: Configurable LLM timeout (default: 60s)
- **Graceful degradation**: Continues processing even if some sentences fail
- **Automatic geo-fallback**: Switches to alternative model if primary is geo-restricted

## Cost Optimization

### Model Costs (per 1M tokens)

**Gemini 3 Pro Preview** (Primary):
- Input: ~$1.25 / Output: ~$5.00
- Best quality and accuracy
- May have geo-restrictions

**Qwen3-VL-8B-Instruct** (Fallback):
- Input: ~$0.10 / Output: ~$0.10
- **90% cheaper than Gemini 3!**
- No geo-restrictions
- Good quality for most tasks
- Direct answers without reasoning overhead

**Recommendation:**
- Use Gemini 3 in supported regions for best quality
- System automatically falls back to Qwen in geo-restricted regions
- Consider using Qwen as primary for cost savings

Estimated costs per document:
- Small document (10 sentences): ~$0.01-0.05 (Gemini) / ~$0.001-0.005 (Qwen)
- Medium document (50 sentences): ~$0.05-0.25 (Gemini) / ~$0.005-0.025 (Qwen)
- Large document (200 sentences): ~$0.20-1.00 (Gemini) / ~$0.02-0.10 (Qwen)

## Troubleshooting

### "No metrics extracted from Excel"
- Check that Excel file contains a proper table with headers
- Ensure numbers are formatted as numbers, not text
- Try a simpler table structure for testing

### "Empty response from vision model"
- Verify OPENROUTER_API_KEY is set correctly
- Check internet connection
- Try increasing timeout in config

### "User location is not supported" (Geo-Restriction)
- **Automatic fix**: System will automatically switch to fallback model
- **Manual fix**: Set Qwen as primary model in config
- **Alternative**: Use Claude models (no geo-restrictions)
- See `AUTOMATIC_FALLBACK_GUIDE.md` for details

### "Confidence too low"
- Lower min_confidence threshold
- Check that Excel data matches document content
- Verify time periods and metric names are consistent

## Examples

See the `examples/` directory for sample files:
- `sample_report.docx`: Example financial report
- `sample_data.xlsx`: Example quarterly data
- `expected_output.docx`: Expected result

## Contributing

Contributions welcome! Areas for improvement:
- Support for more document formats (PDF, Google Docs)
- Multi-sheet Excel support
- Batch processing
- Custom metric name mapping
- Advanced table structure handling

## License

[Your License Here]

