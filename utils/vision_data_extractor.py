"""
Vision-based Data Extractor

Uses vision models to extract structured financial data from Excel images.
"""

import logging
import json
from typing import Dict, Any, Optional
from funnels.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class VisionDataExtractor:
    """Extracts structured data from Excel images using vision models."""
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize the vision data extractor.
        
        Args:
            llm_provider: Optional LLM provider (creates new one if not provided)
        """
        self.llm = llm_provider or LLMProvider()
    
    def extract_financial_data(self, image_base64: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Extract structured financial data from an Excel image.
        
        Args:
            image_base64: Base64 encoded Excel image
            timeout: Timeout in seconds
            
        Returns:
            Dict containing extracted financial metrics and their values
            Format: {
                "metrics": {
                    "Revenue": {
                        "Three Months Ended June 30, 2023": 26930000000,
                        "Six Months Ended June 30, 2023": 42260000000,
                        ...
                    },
                    "Net Income": {...},
                    ...
                },
                "time_periods": ["Three Months Ended June 30, 2023", ...],
                "metadata": {
                    "description": "Brief description of the table",
                    "currency": "USD" or detected currency
                }
            }
        """
        logger.info("Extracting financial data from Excel image using vision model")
        
        prompt = """You are analyzing an Excel spreadsheet image containing financial data.

CRITICAL INSTRUCTIONS:
1. Look at the image carefully and identify:
   - ALL column headers (time periods like "Three Months Ended June 30, 2023")
   - ALL row headers (financial metrics like "Revenue", "Net Income", "Automotive sales", etc.)
   - ALL numerical values in the cells

2. Extract EVERY metric and value you see in the table

3. Output a COMPLETE and VALID JSON object with this EXACT structure:
{
  "metrics": {
    "Metric Name 1": {
      "Time Period 1": 123456789,
      "Time Period 2": 987654321
    },
    "Metric Name 2": {
      "Time Period 1": 111111111,
      "Time Period 2": 222222222
    }
  },
  "time_periods": ["Time Period 1", "Time Period 2"],
  "metadata": {
    "description": "Brief description of the table",
    "currency": "USD"
  }
}

IMPORTANT RULES:
- If you see numbers like "26.93" in a "billions" column, output 26930000000 (multiply by 1,000,000,000)
- If numbers are in millions, multiply by 1,000,000
- If numbers are in thousands, multiply by 1,000
- Use the EXACT metric names from the image
- Use the EXACT time period names from the image
- ALL values must be numbers (integers or floats), NEVER strings
- The "time_periods" array must contain ALL time periods as strings
- DO NOT leave any field empty - if you see data in the image, extract it
- Output ONLY the JSON, no other text before or after

If the image contains a table with data, you MUST extract it. Do not return empty metrics."""

        try:
            # Use max tokens close to model limit (262K total context)
            # Reserve ~5K for input (prompt + image), use rest for output
            result = self.llm.analyze_image(prompt, image_base64, timeout=timeout, max_tokens=257000)
            response_text = result.get("analysis", "")
            
            if not response_text or response_text == "none":
                logger.error("Vision model returned empty response")
                return self._empty_result()
            
            logger.debug(f"Raw vision model response: {response_text[:500]}...")
            
            # Try to parse JSON from response
            # Sometimes models wrap JSON in markdown code blocks
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
            
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            try:
                data = json.loads(response_text)
                logger.info(f"Successfully extracted data for {len(data.get('metrics', {}))} metrics")
                logger.debug(f"Extracted metrics: {list(data.get('metrics', {}).keys())}")
                return data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from vision model response: {e}")
                logger.error(f"Response was: {response_text[:500]}...")
                
                # Try to extract JSON from the response if it's embedded in text
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group(0))
                        logger.info("Successfully extracted JSON from embedded text")
                        return data
                    except:
                        pass
                
                return self._empty_result()
                
        except Exception as e:
            logger.error(f"Error extracting data from image: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return self._empty_result()
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return an empty result structure."""
        return {
            "metrics": {},
            "time_periods": [],
            "metadata": {
                "description": "Failed to extract data",
                "currency": "USD"
            }
        }


def extract_data_from_excel_image(image_base64: str, llm_provider: Optional[LLMProvider] = None,
                                  timeout: int = 60) -> Dict[str, Any]:
    """
    Convenience function to extract data from Excel image.
    
    Args:
        image_base64: Base64 encoded Excel image
        llm_provider: Optional LLM provider
        timeout: Timeout in seconds
        
    Returns:
        Dict containing extracted financial data
    """
    extractor = VisionDataExtractor(llm_provider)
    return extractor.extract_financial_data(image_base64, timeout)

