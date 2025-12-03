"""
Configuration loader from .env file
Replaces config.py with environment-based configuration
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def load_env_file(env_path: Optional[str] = None) -> None:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file. If None, looks for .env in project root.
    """
    if env_path is None:
        # Look for .env in project root (going up from src/galois/)
        current = Path(__file__).resolve()
        
        # Try several common locations
        possible_paths = [
            current.parent.parent.parent / ".env",  # Project root
            current.parent.parent / ".env",          # src/
            current.parent / ".env",                 # src/galois/
            Path.cwd() / ".env",                     # Current working directory
        ]
        
        for path in possible_paths:
            if path.exists():
                env_path = str(path)
                break
    
    if env_path and os.path.exists(env_path):
        print(f"Loading environment from: {env_path}")
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Set environment variable if not already set
                    if key not in os.environ:
                        os.environ[key] = value
    else:
        print("Warning: No .env file found. Using existing environment variables.")


@dataclass
class WatsonxConfig:
    """Configuration for IBM Watsonx"""
    api_key: str
    endpoint: str
    project_id: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 4096
    
    @classmethod
    def from_env(cls) -> WatsonxConfig:
        """Load Watsonx config from environment variables"""
        return cls(
            api_key=os.getenv("WATSONX_API_KEY", ""),
            endpoint=os.getenv("WATSONX_URL", ""),
            project_id=os.getenv("WATSONX_PROJECT_ID", ""),
            model=os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-3-8b-instruct"),
            temperature=float(os.getenv("WATSONX_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("WATSONX_MAX_NEW_TOKENS", "4096")),
        )


@dataclass
class GeminiConfig:
    """Configuration for Google Gemini"""
    api_key: str
    model: str
    temperature: float = 0.0
    max_output_tokens: int = 4096
    
    @classmethod
    def from_env(cls) -> GeminiConfig:
        """Load Gemini config from environment variables"""
        return cls(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.0")),
            max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")),
        )


@dataclass
class AzureConfig:
    """Configuration for Azure OpenAI"""
    endpoint: str
    api_key: str
    api_version: str
    deployment: str
    temperature: float = 0.0
    max_tokens: int = 4096
    
    @classmethod
    def from_env(cls) -> AzureConfig:
        """Load Azure config from environment variables"""
        return cls(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            temperature=float(os.getenv("AZURE_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("AZURE_MAX_TOKENS", "4096")),
        )


@dataclass
class GaloisConfig:
    """Main Galois configuration"""
    llm_provider: str
    watsonx: WatsonxConfig
    gemini: GeminiConfig
    azure: AzureConfig
    
    # Galois-specific settings
    max_iterations: int = 5
    confidence_threshold: float = 0.6
    
    @classmethod
    def from_env(cls, provider: Optional[str] = None, env_path: Optional[str] = None) -> GaloisConfig:
        """
        Load configuration from environment variables.
        
        Args:
            provider: LLM provider to use ("watsonx", "gemini", or "azure")
                     If None, tries to detect from environment
            env_path: Path to .env file. If None, auto-detects.
        
        Returns:
            GaloisConfig instance
        """
        # Load .env file
        load_env_file(env_path)
        
        # Detect provider if not specified
        if provider is None:
            provider = cls._detect_provider()
        
        # Load all provider configs
        watsonx_config = WatsonxConfig.from_env()
        gemini_config = GeminiConfig.from_env()
        azure_config = AzureConfig.from_env()
        
        # Galois settings from env or defaults
        max_iters = int(os.getenv("GALOIS_MAX_ITERATIONS", "5"))
        conf_threshold = float(os.getenv("GALOIS_CONFIDENCE_THRESHOLD", "0.6"))
        
        return cls(
            llm_provider=provider,
            watsonx=watsonx_config,
            gemini=gemini_config,
            azure=azure_config,
            max_iterations=max_iters,
            confidence_threshold=conf_threshold,
        )
    
    @staticmethod
    def _detect_provider() -> str:
        """
        Auto-detect which LLM provider to use based on environment variables.
        
        Priority:
        1. Explicit LLM_PROVIDER env var
        2. Check which provider has API key set
        3. Default to azure
        """
        # Check explicit provider setting
        provider = os.getenv("LLM_PROVIDER")
        if provider:
            return provider.lower()
        
        # Auto-detect based on API keys
        if os.getenv("WATSONX_API_KEY"):
            return "watsonx"
        elif os.getenv("AZURE_OPENAI_API_KEY"):
            return "azure"
        
        elif os.getenv("GEMINI_API_KEY"):
            return "gemini"
        
        # Default
        print("Warning: No LLM provider detected. Defaulting to 'azure'.")
        print("Set LLM_PROVIDER env var to specify explicitly.")
        return "azure"
    
    def validate(self) -> bool:
        """
        Validate that the selected provider has required configuration.
        
        Returns:
            True if valid, raises ValueError if not
        """
        provider = self.llm_provider.lower()
        
        
        
        if provider == "watsonx":
            if not self.watsonx.api_key or not self.watsonx.endpoint:
                raise ValueError(
                    "Watsonx requires WATSONX_API_KEY and WATSONX_URL"
                )
        
        elif provider == "azure":
            if not self.azure.endpoint or not self.azure.api_key:
                raise ValueError(
                    "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
                )
        
        elif provider == "gemini":
            if not self.gemini.api_key:
                raise ValueError(
                    "Gemini requires GEMINI_API_KEY"
                )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        return True


# Convenience function
def load_config(provider: Optional[str] = None, env_path: Optional[str] = None) -> GaloisConfig:
    """
    Load Galois configuration from .env file.
    
    Args:
        provider: LLM provider ("watsonx", "gemini", "azure"). Auto-detected if None.
        env_path: Path to .env file. Auto-detected if None.
    
    Returns:
        GaloisConfig instance
    
    Example:
        # Auto-detect everything
        config = load_config()
        
        # Specify provider
        config = load_config(provider="azure")
        
        # Specify .env location
        config = load_config(env_path="/path/to/.env")
    """
    config = GaloisConfig.from_env(provider=provider, env_path=env_path)
    config.validate()
    return config


# Default config instance (lazy loaded)
_default_config: Optional[GaloisConfig] = None


def get_config() -> GaloisConfig:
    """
    Get the default configuration instance (singleton).
    
    Returns:
        GaloisConfig instance
    """
    global _default_config
    if _default_config is None:
        _default_config = load_config()
    return _default_config