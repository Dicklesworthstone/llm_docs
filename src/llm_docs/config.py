"""
Configuration management for llm_docs.
"""

import configparser
import os
from pathlib import Path
from typing import Any, Dict, Optional

from decouple import Config as DecoupleConfig
from decouple import RepositoryEnv
from pydantic import BaseModel, Field

# Initialize decouple config
# First try to load from .env file, fall back to environment variables
env_file = Path('.env')
if env_file.exists():
    config_decouple = DecoupleConfig(RepositoryEnv('.env'))
else:
    from decouple import config as config_decouple


class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str = Field(default="sqlite:///llm_docs.db")
    echo: bool = Field(default=False)


class APIConfig(BaseModel):
    """API configuration."""
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    cors_origins: list[str] = Field(default=["*"])
    enable_auth: bool = Field(default=False)


class LLMProviderConfig(BaseModel):
    """Configuration for a specific LLM provider."""
    provider: str = Field(default="anthropic")  # anthropic, openai, google, etc.
    model: str = Field(default="claude-3-7-sonnet-20250219")  # Model name (without provider prefix)
    api_key: Optional[str] = Field(default=None)  # API key for this provider
    max_tokens: int = Field(default=4000)
    temperature: float = Field(default=0.1)


class LLMConfig(BaseModel):
    """Configuration for LLM usage across different parts of the system."""
    # Default provider to use if a specific provider isn't specified
    default: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    
    # Provider for distillation
    distillation: Optional[LLMProviderConfig] = None
    
    # Provider for browser exploration
    browser_exploration: Optional[LLMProviderConfig] = None
    
    # Provider for documentation extraction
    doc_extraction: Optional[LLMProviderConfig] = None


class ProcessingConfig(BaseModel):
    """Processing configuration."""
    docs_dir: str = Field(default="docs")
    distilled_docs_dir: str = Field(default="distilled_docs")
    max_chunk_tokens: int = Field(default=80000)
    max_pages_per_doc: int = Field(default=100)
    doc_chunk_count: int = Field(default=5)
    parallelism: int = Field(default=1)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO")
    file: Optional[str] = Field(default=None)
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class Config(BaseModel):
    """Main configuration."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)  # LLM configuration for multiple providers
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# Directories
CONFIG_DIRS = [
    Path.cwd(),
    Path.home() / ".config" / "llm_docs",
    Path("/etc/llm_docs")
]
CONFIG_FILENAME = "llm_docs.conf"


def load_config_from_file(path: Path) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        path: Path to the configuration file
        
    Returns:
        Dictionary of configuration values
    """
    config = configparser.ConfigParser()
    config.read(path)
    
    # Convert to dictionary
    result = {}
    for section in config.sections():
        # Handle nested sections (e.g., llm.default)
        if "." in section:
            parent, child = section.split(".", 1)
            if parent not in result:
                result[parent] = {}
            if isinstance(result[parent], dict):  # Ensure it's a dict
                result[parent][child] = {}
                for key, value in config[section].items():
                    # Try to parse values
                    result[parent][child][key] = parse_config_value(value)
        else:
            result[section] = {}
            for key, value in config[section].items():
                # Try to parse values
                result[section][key] = parse_config_value(value)
                
    return result


def parse_config_value(value: str) -> Any:
    """Parse a configuration value into the appropriate type."""
    try:
        # Handle boolean values
        if value.lower() in ("true", "yes", "1"):
            return True
        elif value.lower() in ("false", "no", "0"):
            return False
        # Handle numeric values
        elif value.isdigit():
            return int(value)
        elif value.replace(".", "", 1).isdigit():
            return float(value)
        # Handle lists
        elif "," in value:
            return [v.strip() for v in value.split(",")]
        # Otherwise, keep as string
        else:
            return value
    except Exception:
        # Fall back to string if parsing fails
        return value


def load_config() -> Config:
    """
    Load configuration from files and environment variables.
    
    Returns:
        Config object
    """
    # Start with default config
    config_dict = {}
    
    # Find and load config file
    for config_dir in CONFIG_DIRS:
        config_path = config_dir / CONFIG_FILENAME
        if config_path.exists():
            config_dict.update(load_config_from_file(config_path))
            break
    
    # Override with environment variables using python-decouple
    # Format: LLM_DOCS__SECTION__KEY or LLM_DOCS__SECTION__SUBSECTION__KEY
    # First, prepare a list of expected environment variables
    expected_env_vars = []
    
    # Add standard config patterns
    for section in ["database", "api", "llm", "processing", "logging"]:
        for key in ["url", "host", "port", "level", "file", "format", "echo", "enable_auth", "docs_dir", "distilled_docs_dir"]:
            expected_env_vars.append(f"LLM_DOCS__{section.upper()}__{key.upper()}")
    
    # Add LLM config patterns for each provider section
    for section in ["default", "distillation", "browser_exploration", "doc_extraction"]:
        for key in ["provider", "model", "api_key", "max_tokens", "temperature"]:
            expected_env_vars.append(f"LLM_DOCS__LLM__{section.upper()}__{key.upper()}")
    
    # Check each expected environment variable
    for env_var in expected_env_vars:
        try:
            value = config_decouple(env_var, default=None)
            if value is not None:
                parts = env_var.split("__")
                
                if len(parts) == 3:
                    # Simple section.key format
                    _, section, key = parts
                    section = section.lower()
                    key = key.lower()
                    
                    if section not in config_dict:
                        config_dict[section] = {}
                    
                    # Parse the value
                    config_dict[section][key] = parse_config_value(value)
                        
                elif len(parts) == 4:
                    # Nested section.subsection.key format
                    _, section, subsection, key = parts
                    section = section.lower()
                    subsection = subsection.lower()
                    key = key.lower()
                    
                    if section not in config_dict:
                        config_dict[section] = {}
                    
                    if subsection not in config_dict[section]:
                        config_dict[section][subsection] = {}
                    
                    # Parse the value
                    config_dict[section][subsection][key] = parse_config_value(value)
        except Exception:
            # If decouple fails, fall back to os.environ for backward compatibility
            pass
    
    # Fall back to direct environment variables for backward compatibility
    for key, value in os.environ.items():
        if key.startswith("LLM_DOCS__"):
            parts = key.split("__")
            
            if len(parts) == 3:
                # Simple section.key format
                _, section, key = parts
                section = section.lower()
                key = key.lower()
                
                if section not in config_dict:
                    config_dict[section] = {}
                
                # Parse the value
                config_dict[section][key] = parse_config_value(value)
                    
            elif len(parts) == 4:
                # Nested section.subsection.key format
                _, section, subsection, key = parts
                section = section.lower()
                subsection = subsection.lower()
                key = key.lower()
                
                if section not in config_dict:
                    config_dict[section] = {}
                
                if subsection not in config_dict[section]:
                    config_dict[section][subsection] = {}
                
                # Parse the value
                config_dict[section][subsection][key] = parse_config_value(value)
    
    # Check for API keys from environment variables
    # Support all providers
    provider_env_vars = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "groq": "GROQ_API_KEY",
        "sambanova": "SAMBANOVA_API_KEY",
        "watsonx": "WATSONX_API_KEY",
        "ollama": "OLLAMA_API_KEY"
    }
    
    # Add llm section if it doesn't exist
    if "llm" not in config_dict:
        config_dict["llm"] = {}
        
    # Add default section if it doesn't exist
    if "default" not in config_dict["llm"]:
        config_dict["llm"]["default"] = {}
    
    # Check each provider API key using decouple
    for provider, env_var in provider_env_vars.items():
        try:
            # Try to get the API key from decouple
            api_key = config_decouple(env_var, default=None)
            
            if api_key:
                # If this is the provider in default config, add the API key
                if config_dict["llm"]["default"].get("provider") == provider:
                    config_dict["llm"]["default"]["api_key"] = api_key
                
                # Also check other sections
                for section in ["distillation", "browser_exploration", "doc_extraction"]:
                    if section in config_dict["llm"] and config_dict["llm"][section].get("provider") == provider:
                        config_dict["llm"][section]["api_key"] = api_key
        except Exception:
            # Fall back to checking environment variables directly for backward compatibility
            if env_var in os.environ:
                # If this is the provider in default config, add the API key
                if config_dict["llm"]["default"].get("provider") == provider:
                    config_dict["llm"]["default"]["api_key"] = os.environ[env_var]
                
                # Also check other sections
                for section in ["distillation", "browser_exploration", "doc_extraction"]:
                    if section in config_dict["llm"] and config_dict["llm"][section].get("provider") == provider:
                        config_dict["llm"][section]["api_key"] = os.environ[env_var]
        
    # Create the config object
    return Config(**config_dict)


# Global config instance
config = load_config()