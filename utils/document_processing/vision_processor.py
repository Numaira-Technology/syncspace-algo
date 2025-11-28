"""
Vision-Based Document Processor

Uses vision AI to understand Excel tables and update Word documents intelligently.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from docx import Document
from docx.shared import Pt, RGBColor
from config import Config
from funnels.extract import read_docx
from funnels.llm_provider import get_llm_provider
from utils.excel_to_image import convert_excel_to_image
from utils.vision_data_extractor import extract_data_from_excel_image
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


def analyze_and_update_document(docx_path: str, excel_image_base64: str, 
                                extracted_data: Dict[str, Any], 
                                timeout: int = None) -> List[Tuple[str, str, float, str]]:
    """
    Analyze Word document and determine what needs to be updated based on Excel data.
    
    Args:
        docx_path: Path to Word document
        excel_image_base64: Base64 encoded Excel image
        extracted_data: Structured data extracted from Excel
        timeout: Optional timeout for LLM calls
        
    Returns:
        List of tuples: (original_sentence, updated_sentence, confidence, description)
    """
    logger.info("Starting vision-based document analysis")
    
    # Get LLM provider
    config = Config()
    llm_config = config.get_model_config()["llm"]
    timeout = timeout or llm_config.get("timeout", 30)
    llm = get_llm_provider()
    
    # Read Word document
    logger.info(f"Reading Word document: {docx_path}")
    sentences = read_docx(docx_path)
    logger.info(f"Found {len(sentences)} sentences in document")
    
    if not sentences:
        logger.warning("No sentences found in document")
        return []
    
    # Prepare extracted data summary for context
    metrics_list = list(extracted_data.get("metrics", {}).keys())
    time_periods = extracted_data.get("time_periods", [])
    
    logger.info(f"Available metrics: {metrics_list}")
    logger.info(f"Available time periods: {time_periods}")
    
    # Process each sentence to find ones that need updating
    results = []
    print("\nAnalyzing document with vision AI...")
    
    for sentence in tqdm(sentences.values(), desc="Analyzing", unit="sentence"):
        if len(sentence.strip()) < 20:  # Skip very short sentences
            logger.debug(f"Skipping short sentence: '{sentence[:30]}...'")
            continue
        
        # Ask AI to analyze if this sentence contains financial data that should be updated
        analysis_prompt = f"""Analyze this sentence from a financial document:

Sentence: "{sentence}"

Available financial data from Excel:
- Metrics: {', '.join(metrics_list)}
- Time periods: {', '.join(time_periods)}

Task: Determine if this sentence contains financial data that should be updated with values from the Excel data.

Output a JSON object with this format:
{{
  "should_update": true/false,
  "reason": "Brief explanation",
  "metrics_mentioned": ["list", "of", "metrics", "if", "any"],
  "time_periods_mentioned": ["list", "of", "periods", "if", "any"]
}}

Only output the JSON, nothing else."""

        try:
            analysis_result = llm.analyze_text(analysis_prompt, timeout=timeout, max_tokens=500)
            analysis_text = analysis_result.get("analysis", "")
            
            if not analysis_text or analysis_text == "none":
                logger.debug(f"No analysis for sentence: '{sentence[:50]}...'")
                continue
            
            # Parse analysis
            analysis_text = analysis_text.strip()
            if analysis_text.startswith("```json"):
                analysis_text = analysis_text[7:]
            elif analysis_text.startswith("```"):
                analysis_text = analysis_text[3:]
            if analysis_text.endswith("```"):
                analysis_text = analysis_text[:-3]
            analysis_text = analysis_text.strip()
            
            try:
                analysis = json.loads(analysis_text)
            except json.JSONDecodeError:
                # Try to extract JSON
                import re
                json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(0))
                else:
                    logger.debug(f"Could not parse analysis JSON: {analysis_text[:100]}...")
                    continue
            
            if not analysis.get("should_update", False):
                logger.debug(f"Sentence doesn't need update: '{sentence[:50]}...' - {analysis.get('reason', 'No reason')}")
                continue
            
            logger.info(f"Found sentence to update: '{sentence[:50]}...'")
            logger.info(f"  Metrics: {analysis.get('metrics_mentioned', [])}")
            logger.info(f"  Periods: {analysis.get('time_periods_mentioned', [])}")
            
            # Now ask AI to update the sentence with the correct data
            update_prompt = f"""Update this sentence with the correct financial data from the Excel table.

Original Sentence: "{sentence}"

Excel Data (JSON format):
{json.dumps(extracted_data, indent=2)}

Instructions:
1. Identify which metrics and time periods are mentioned in the sentence
2. Look up the correct values from the Excel data
3. Update ONLY the numerical values in the sentence
4. Keep the exact same sentence structure and wording
5. Maintain the same number formatting style (e.g., if it says "$X.XX billion", keep that format)
6. Do NOT change any other text

Output ONLY the updated sentence, nothing else. If you cannot update it confidently, output: SKIP"""

            update_result = llm.analyze_text(update_prompt, timeout=timeout, max_tokens=500)
            updated_text = update_result.get("analysis", "").strip()
            
            if not updated_text or updated_text == "none" or updated_text == "SKIP":
                logger.warning(f"Could not generate update for: '{sentence[:50]}...'")
                continue
            
            # Verify the update is different and reasonable
            if updated_text == sentence:
                logger.debug(f"Update identical to original, skipping: '{sentence[:50]}...'")
                continue
            
            # Calculate confidence based on:
            # - Sentence length similarity (should be very close)
            # - Presence of mentioned metrics
            # - Structural similarity
            len_ratio = len(updated_text) / len(sentence) if len(sentence) > 0 else 0
            len_similarity = 1.0 - abs(1.0 - len_ratio)  # Closer to 1.0 is better
            
            # Check if metrics are still present
            metrics_present = sum(1 for m in analysis.get('metrics_mentioned', []) 
                                 if m.lower() in updated_text.lower())
            metric_score = metrics_present / max(len(analysis.get('metrics_mentioned', [])), 1)
            
            # Overall confidence
            confidence = 0.6 + (len_similarity * 0.2) + (metric_score * 0.2)
            confidence = min(confidence, 0.99)  # Cap at 0.99
            
            description = f"Updated {', '.join(analysis.get('metrics_mentioned', []))} for {', '.join(analysis.get('time_periods_mentioned', []))}"
            
            logger.info(f"Successfully updated sentence (confidence: {confidence:.2f})")
            logger.info(f"  Original: '{sentence[:80]}...'")
            logger.info(f"  Updated:  '{updated_text[:80]}...'")
            
            results.append((sentence, updated_text, confidence, description))
            
        except Exception as e:
            logger.error(f"Error processing sentence: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            continue
    
    logger.info(f"Analysis complete. Found {len(results)} sentences to update")
    return results


def process_files_with_vision(docx_path: str, excel_path: str, timeout: int = None) -> List[Tuple[str, str, float, str]]:
    """
    Process Word and Excel files using vision-based approach.
    
    Args:
        docx_path: Path to Word document
        excel_path: Path to Excel file
        timeout: Optional timeout for LLM calls
        
    Returns:
        List of tuples: (original_sentence, updated_sentence, confidence, description)
    """
    logger.info("=" * 80)
    logger.info("VISION-BASED DOCUMENT PROCESSING")
    logger.info("=" * 80)
    
    try:
        # Step 1: Convert Excel to image
        logger.info("Step 1: Converting Excel to image...")
        img, img_base64 = convert_excel_to_image(excel_path)
        logger.info(f"Excel converted to image ({img.width}x{img.height} pixels)")
        
        # Step 2: Extract structured data from image
        logger.info("Step 2: Extracting data from Excel image using vision AI...")
        extracted_data = extract_data_from_excel_image(img_base64, timeout=timeout)
        
        if not extracted_data.get("metrics"):
            logger.error("Failed to extract any metrics from Excel image")
            return []
        
        logger.info(f"Successfully extracted {len(extracted_data['metrics'])} metrics")
        logger.info(f"Metrics: {list(extracted_data['metrics'].keys())}")
        
        # Step 3: Analyze Word document and determine updates
        logger.info("Step 3: Analyzing Word document and determining updates...")
        results = analyze_and_update_document(docx_path, img_base64, extracted_data, timeout)
        
        logger.info("=" * 80)
        logger.info(f"PROCESSING COMPLETE: {len(results)} updates identified")
        logger.info("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in vision-based processing: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def apply_updates_to_document(docx_path: str, output_path: str, 
                              updates: List[Tuple[str, str, float, str]],
                              min_confidence: float = 0.7) -> int:
    """
    Apply updates to Word document with highlighting.
    
    Args:
        docx_path: Path to original Word document
        output_path: Path to save updated document
        updates: List of (original, updated, confidence, description) tuples
        min_confidence: Minimum confidence to apply update
        
    Returns:
        Number of updates applied
    """
    logger.info(f"Applying updates to document (min confidence: {min_confidence})")
    
    doc = Document(docx_path)
    updates_applied = 0
    
    # Create a mapping of original to updated text
    update_map = {orig: (upd, conf, desc) for orig, upd, conf, desc in updates if conf >= min_confidence}
    
    logger.info(f"Total updates to apply: {len(update_map)}")
    
    for paragraph in doc.paragraphs:
        para_text = paragraph.text
        
        # Check if this paragraph contains any text to update
        for original_text, (updated_text, confidence, description) in update_map.items():
            if original_text in para_text:
                logger.info(f"Applying update in paragraph (confidence: {confidence:.2f})")
                logger.debug(f"  Original: '{original_text[:80]}...'")
                logger.debug(f"  Updated:  '{updated_text[:80]}...'")
                
                # Replace the text
                para_text = para_text.replace(original_text, updated_text)
                
                # Clear existing runs and create new one with highlighting
                paragraph.clear()
                run = paragraph.add_run(para_text)
                
                # Highlight the updated text in yellow
                run.font.highlight_color = 7  # Yellow
                
                updates_applied += 1
                break  # Only apply one update per paragraph
    
    # Save the document
    doc.save(output_path)
    logger.info(f"Document saved to: {output_path}")
    logger.info(f"Total updates applied: {updates_applied}")
    
    return updates_applied

