from typing import Dict, List
from pathlib import Path

class BaseConfig:
    # Environment settings
    ENV_VARS = {
        "TOKENIZERS_PARALLELISM": "false",
        "HF_ENDPOINT": "https://hf-mirror.com"
    }
    
    # Model configurations
    MODEL_CONFIG = {
        "embedding": {
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "similarity_threshold": 0.4
        },
        "llm": {
            "provider": "openrouter",
            "model": "google/gemini-3-pro-preview",
            "timeout": 30
        }
    }
    
    # Confidence scoring weights
    CONFIDENCE_WEIGHTS = {
        "embedding_similarity": 0.3,  # Initial embedding match
        "llm_verification": 0.4,      # LLM-based verification
        "temporal_context": 0.3,      # Temporal context analysis
    }
    
    # File processing
    FILE_PROCESSING = {
        "progress_bar": {
            "desc": "Processing",
            "ncols": 100,
            "position": 0,
            "leave": True
        }
    }
    
    @classmethod
    def get_env_vars(cls) -> Dict[str, str]:
        return cls.ENV_VARS
    
    @classmethod
    def get_model_config(cls) -> Dict:
        return cls.MODEL_CONFIG
    
    @classmethod
    def get_confidence_weights(cls) -> Dict[str, float]:
        return cls.CONFIDENCE_WEIGHTS
    
    @classmethod
    def get_file_processing_config(cls) -> Dict:
        return cls.FILE_PROCESSING 