from typing import List, Dict, Tuple, Optional, Any
import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx
import logging
from config import Config

load_dotenv()

logger = logging.getLogger(__name__)

class OpenRouterProvider:
    """OpenRouter provider using OpenAI SDK with OpenRouter base URL."""
    
    def __init__(self, model: str, timeout: int = 30, vision_model: Optional[str] = None):
        """Initialize OpenRouter provider.
        
        Args:
            model: Text-only model identifier (e.g., "qwen/qwen-plus")
            timeout: Default timeout in seconds
            vision_model: Vision model for image OCR tasks (e.g., "qwen/qwen-vl-max")
        """
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not found")
        
        self.model = model
        self.vision_model = vision_model
        self.default_timeout = timeout
        
        # Create a custom httpx client without proxies parameter
        # This avoids compatibility issues with httpx 0.28+
        http_client = httpx.Client(
            timeout=httpx.Timeout(timeout=timeout, connect=5.0)
        )
        
        # Initialize OpenAI client with custom http_client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            http_client=http_client
        )
        
        logger.info(f"Initialized OpenRouter with text model: {model}")
        if vision_model:
            logger.info(f"Vision model configured: {vision_model}")
    
    def _call_openrouter(self, messages: list, max_tokens: int = 200, timeout: Optional[int] = None,
                        image_base64: Optional[str] = None) -> str:
        """Call OpenRouter API with proper error handling.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            timeout: Timeout in seconds (defaults to self.default_timeout)
            image_base64: Optional base64 encoded image; triggers use of vision_model
            
        Returns:
            Generated text or "none" on error/timeout
        """
        if timeout is None:
            timeout = self.default_timeout
        
        # If image is provided, format messages for vision model
        if image_base64:
            # Convert text-only messages to vision format
            vision_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    vision_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": msg["content"]
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    })
                else:
                    vision_messages.append(msg)
            messages = vision_messages
            logger.debug("Formatted messages for vision model with image")
        
        # Select model: use vision_model for image tasks, text model otherwise
        if image_base64 and self.vision_model:
            model_to_use = self.vision_model
            logger.info(f"Vision task — using vision model: {model_to_use}")
        else:
            model_to_use = self.model
            if image_base64:
                logger.warning("Vision task but no vision_model configured — falling back to text model")
            logger.info(f"Text task — using model: {model_to_use}")
        
        try:
            logger.debug(f"Calling OpenRouter API with model {model_to_use}")
            if not image_base64:
                logger.debug(f"Messages being sent: {messages}")
            else:
                logger.debug(f"Messages with image (image size: {len(image_base64)} bytes)")
            logger.debug(f"Max tokens: {max_tokens}, Temperature: 0, Timeout: {timeout}")
            
            # Check if this is a reasoning model (Gemini 3, DeepSeek, or Qwen Thinking)
            # NOTE: For vision tasks, we should have already switched to a non-reasoning model above
            extra_body = {}
            model_lower_check = model_to_use.lower()
            
            # Enable reasoning mode for text-only tasks with reasoning models
            if not image_base64 and ("gemini-3" in model_lower_check or "deepseek" in model_lower_check or "thinking" in model_lower_check):
                extra_body = {"reasoning": {"enabled": True}}
                logger.debug(f"Enabling reasoning mode for text-only model: {model_to_use}")
            elif image_base64:
                logger.debug(f"Vision task - using model without reasoning mode: {model_to_use}")
            
            # Use the single client instance - timeout is handled by the API call itself
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
                timeout=timeout,  # Pass timeout to the API call, not the client
                extra_headers={
                    "HTTP-Referer": "https://github.com/syncspace-algo",
                    "X-Title": "SyncSpace Algo"
                },
                extra_body=extra_body if extra_body else None
            )
            
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                logger.debug(f"Finish reason: {choice.finish_reason}")
                
                text = choice.message.content
                
                # For reasoning models, check if reasoning_details are present
                if hasattr(choice.message, 'reasoning_details') and choice.message.reasoning_details:
                    logger.debug(f"Reasoning model detected - has reasoning_details")
                    logger.debug(f"Reasoning details: {str(choice.message.reasoning_details)[:200]}...")
                
                # Check if there's a 'reasoning' field (some models put response here)
                if hasattr(choice.message, 'reasoning') and choice.message.reasoning:
                    logger.debug(f"Model has 'reasoning' field: {str(choice.message.reasoning)[:200]}...")
                
                if text is None or text == "":
                    logger.warning(f"Empty or invalid response from API: content is None or empty")
                    logger.warning(f"Finish reason: {choice.finish_reason}")
                    
                    # Try to extract from reasoning field if available
                    if hasattr(choice.message, 'reasoning') and choice.message.reasoning:
                        logger.info("Content is empty, but 'reasoning' field has data - this shouldn't happen for vision tasks!")
                        logger.info("The model spent all tokens on reasoning instead of the actual answer.")
                        logger.info("This is likely because reasoning mode was enabled for a vision task.")
                        # Don't use reasoning as the answer - it's not the structured data we need
                    
                    # Check if there's a refusal or other message fields
                    if hasattr(choice.message, 'refusal') and choice.message.refusal:
                        logger.error(f"Model refused to respond: {choice.message.refusal}")
                    
                    return "none"
                
                # Log the raw response before any processing
                logger.debug(f"Raw API response content: '{text}'")
                logger.debug(f"Response length: {len(text)} characters")
                    
                text = text.strip()
                logger.debug(f"After strip: '{text}' (length: {len(text)})")
                
                if text:
                    logger.debug(f"Processed response text: {text[:100]}...")
                    return text
                else:
                    logger.warning(f"Empty or invalid response from API: text is empty after cleanup")
                    logger.warning(f"Original content was: '{response.choices[0].message.content}'")
                    return "none"
            
            logger.warning(f"Empty or invalid response from API: no choices (got {len(response.choices) if response.choices else 0} choices)")
            return "none"
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error calling OpenRouter API: {error_msg}")
            
            # Check for specific error types
            if "404" in error_msg and "No endpoints found" in error_msg:
                logger.error(f"Model '{model_to_use}' not found on OpenRouter.")
                logger.error("Common valid models include:")
                logger.error("  - google/gemini-3-pro-preview (best quality, may have geo-restrictions)")
                logger.error("  - qwen/qwen3-vl-8b-thinking (good alternative, no geo-restrictions)")
                logger.error("  - anthropic/claude-3.5-sonnet:beta (no geo-restrictions)")
                logger.error("Please check https://openrouter.ai/models for available models")
            
            elif ("User location is not supported" in error_msg or "FAILED_PRECONDITION" in error_msg
                  or ("403" in error_msg and "not available in your region" in error_msg)):
                logger.error(
                    f"GEO-RESTRICTION: Model '{model_to_use}' is not available in your region. "
                    "Update model/vision_model in config/base_config.py to a non-geo-restricted model."
                )
            
            elif "400" in error_msg or "Bad Request" in error_msg:
                logger.error("Bad Request - possible causes:")
                logger.error("  1. Model not available in your region")
                logger.error("  2. Invalid request format")
                logger.error("  3. Image too large (try smaller Excel files)")
                logger.error(f"  Full error: {error_msg[:500]}")
            
            import traceback
            logger.debug(traceback.format_exc())
            return "none"
    
    def analyze_text(self, text: str, timeout: Optional[int] = None, max_tokens: int = 200) -> Dict[str, Any]:
        """Analyze text using OpenRouter.
        
        Args:
            text: Text to analyze
            timeout: Optional timeout in seconds
            max_tokens: Maximum tokens for the response (default: 200)
            
        Returns:
            Dict with analysis results
        """
        messages = [
            {
                "role": "user",
                "content": text
            }
        ]
        response = self._call_openrouter(messages, max_tokens=max_tokens, timeout=timeout)
        return {"analysis": response} if response else {}
    
    def analyze_image(self, prompt: str, image_base64: str, timeout: Optional[int] = None, 
                     max_tokens: int = 4096) -> Dict[str, Any]:
        """Analyze an image using vision model.
        
        Args:
            prompt: Text prompt describing what to analyze
            image_base64: Base64 encoded image
            timeout: Optional timeout in seconds
            max_tokens: Maximum tokens for output
            
        Returns:
            Dict with analysis results
        """
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        response = self._call_openrouter(messages, max_tokens=max_tokens, timeout=timeout, 
                                        image_base64=image_base64)
        return {"analysis": response} if response else {}
        
    def compare_text(self, text1: str, text2: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Compare two pieces of text using OpenRouter.
        
        Args:
            text1: First text
            text2: Second text
            timeout: Optional timeout in seconds
            
        Returns:
            Dict with comparison results including confidence score
        """
        prompt = f"""Compare these two pieces of text semantically:

Text 1: {text1}
Text 2: {text2}

Are they referring to the same thing? Output format:
{{"match": true/false, "confidence": 0.0-1.0}}"""

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        response = self._call_openrouter(messages, max_tokens=200, timeout=timeout)
        return response if response and response != "none" else {"match": False, "confidence": 0.0}
        
    def update_text(self, text: str, metric: str, values: Dict[str, float], timeout: Optional[int] = None) -> str:
        """Update text with new values using OpenRouter.
        
        Args:
            text: Original text
            metric: Metric to update
            values: New values to insert
            timeout: Optional timeout in seconds
            
        Returns:
            Updated text
        """
        values_str = ", ".join(f"{k}: {v}" for k, v in values.items())
        prompt = f"""Update this text by replacing the values for the metric "{metric}" with these new values:
{values_str}

Text: {text}

Output only the updated text."""

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        response = self._call_openrouter(messages, max_tokens=200, timeout=timeout)
        return response if response and response != "none" else text

class LLMProvider:
    """Wrapper class for LLM provider that uses OpenRouter."""
    
    def __init__(self, provider: str = None):
        config = Config()
        llm_config = config.get_model_config()["llm"]
        self.provider = provider or llm_config["provider"]
        self.model = llm_config["model"]
        self.vision_model = llm_config.get("vision_model")
        self.default_timeout = llm_config["timeout"]
        
        if self.provider == "openrouter":
            self._provider = OpenRouterProvider(
                self.model,
                self.default_timeout,
                vision_model=self.vision_model
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Only 'openrouter' is supported.")
    
    def analyze_text(self, text: str, timeout: Optional[int] = None, max_tokens: int = 200) -> Dict[str, Any]:
        """Analyze text using the configured provider."""
        if timeout is None:
            timeout = self.default_timeout
        return self._provider.analyze_text(text, timeout, max_tokens)
    
    def analyze_image(self, prompt: str, image_base64: str, timeout: Optional[int] = None, 
                     max_tokens: int = 1000) -> Dict[str, Any]:
        """Analyze an image using vision model.
        
        Args:
            prompt: Text prompt describing what to analyze
            image_base64: Base64 encoded image
            timeout: Optional timeout in seconds
            max_tokens: Maximum tokens for the response
            
        Returns:
            Dict with analysis results
        """
        if timeout is None:
            timeout = self.default_timeout
        return self._provider.analyze_image(prompt, image_base64, timeout, max_tokens)
    
    def compare_text(self, text1: str, text2: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Compare two pieces of text using the configured provider."""
        if timeout is None:
            timeout = self.default_timeout
        return self._provider.compare_text(text1, text2, timeout)
    
    def update_text(self, text: str, metric: str, values: Dict[str, float], timeout: Optional[int] = None) -> str:
        """Update text with new values using the configured provider."""
        if timeout is None:
            timeout = self.default_timeout
        return self._provider.update_text(text, metric, values, timeout)
    
    def batch_check_metrics(
        self,
        sentence: str,
        target_metrics: List[str],
        timeout: int = None
    ) -> List[str]:
        """Check if target metrics are present in the sentence.
        
        Args:
            sentence: The sentence to check
            target_metrics: List of metric names to look for
            timeout: Optional timeout in seconds (defaults to config value)
            
        Returns:
            List of metrics found in the sentence
        """
        if timeout is None:
            timeout = self.default_timeout
            
        prompt = f"""Check which metrics from this list are present in the sentence:

Sentence: {sentence}

Target Metrics:
{chr(10).join(f"- {metric}" for metric in target_metrics)}

Output only the matching metrics as a comma-separated list. If no matches, output: none"""

        try:
            logger.debug(f"batch_check_metrics called with sentence: '{sentence[:100]}...'")
            logger.debug(f"Target metrics: {target_metrics}")
            
            result = self._provider.analyze_text(prompt, timeout)
            logger.debug(f"analyze_text returned: {result}")
            
            text_response = result.get("analysis", "none").strip().lower()
            logger.debug(f"text_response after processing: '{text_response}'")
            
            if text_response == "none":
                logger.debug("No metrics found (response was 'none')")
                return []
                
            matched_metrics = [m.strip() for m in text_response.split(",")]
            logger.debug(f"Matched metrics: {matched_metrics}")
            return matched_metrics
            
        except Exception as e:
            logger.error(f"Error in batch_check_metrics: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

def get_llm_provider(provider: str = None) -> LLMProvider:
    """Get an LLM provider instance.
    
    Args:
        provider: Optional provider name (defaults to config value)
        
    Returns:
        LLMProvider instance
    """
    return LLMProvider(provider)                  