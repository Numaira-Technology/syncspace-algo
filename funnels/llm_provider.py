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
    
    def __init__(self, model: str, timeout: int = 30):
        """Initialize OpenRouter provider.
        
        Args:
            model: Model identifier (e.g., "google/gemini-3-pro-preview")
            timeout: Default timeout in seconds
        """
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not found")
        
        self.model = model
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
    
    def _call_openrouter(self, messages: list, max_tokens: int = 200, timeout: Optional[int] = None) -> str:
        """Call OpenRouter API with proper error handling and logging.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            timeout: Timeout in seconds (defaults to self.default_timeout)
            
        Returns:
            Generated text or "none" on error/timeout
        """
        if timeout is None:
            timeout = self.default_timeout
        
        try:
            logger.debug(f"Calling OpenRouter API with model {self.model}")
            logger.debug(f"Messages being sent: {messages}")
            logger.debug(f"Max tokens: {max_tokens}, Temperature: 0, Timeout: {timeout}")
            
            # Check if this is a reasoning model (Gemini 3 Pro Preview or DeepSeek)
            extra_body = {}
            if "gemini-3-pro-preview" in self.model.lower() or "deepseek" in self.model.lower():
                extra_body = {"reasoning": {"enabled": True}}
                logger.debug(f"Enabling reasoning mode for model: {self.model}")
            
            # Use the single client instance - timeout is handled by the API call itself
            response = self.client.chat.completions.create(
                model=self.model,
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
                
                if text is None or text == "":
                    logger.warning(f"Empty or invalid response from API: content is None or empty")
                    logger.warning(f"Finish reason: {choice.finish_reason}")
                    logger.warning(f"Message object: {choice.message}")
                    
                    # Check if there's a refusal or other message fields
                    if hasattr(choice.message, 'refusal') and choice.message.refusal:
                        logger.error(f"Model refused to respond: {choice.message.refusal}")
                    
                    return "none"
                
                # Log the raw response before any processing
                logger.debug(f"Raw API response content: '{text}'")
                logger.debug(f"Response length: {len(text)} characters")
                    
                text = text.strip()
                logger.debug(f"After strip: '{text}' (length: {len(text)})")
                
                # Clean up the response by removing [] and extra whitespace
                text = text.replace('[]', '').strip()
                logger.debug(f"After cleanup: '{text}' (length: {len(text)})")
                
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
            
            # If it's a 404 model not found error, provide helpful suggestions
            if "404" in error_msg and "No endpoints found" in error_msg:
                logger.error(f"Model '{self.model}' not found on OpenRouter.")
                logger.error("Common valid models include:")
                logger.error("  - google/gemini-3-pro-preview (reasoning model)")
                logger.error("  - deepseek/deepseek-chat-v3.1 (reasoning model, fallback)")
                logger.error("  - google/gemini-pro-1.5")
                logger.error("  - anthropic/claude-3-haiku")
                logger.error("  - meta-llama/llama-3.1-8b-instruct:free")
                logger.error("Please check https://openrouter.ai/models for available models")
            
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
        return {"analysis": response} if response and response != "none" else {}
        
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
        self.default_timeout = llm_config["timeout"]
        
        if self.provider == "openrouter":
            self._provider = OpenRouterProvider(self.model, self.default_timeout)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Only 'openrouter' is supported.")
    
    def analyze_text(self, text: str, timeout: Optional[int] = None, max_tokens: int = 200) -> Dict[str, Any]:
        """Analyze text using the configured provider."""
        if timeout is None:
            timeout = self.default_timeout
        return self._provider.analyze_text(text, timeout, max_tokens)
    
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