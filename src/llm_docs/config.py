"""
Configuration management for llm_docs.
"""

import configparser
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


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


class AnthropicConfig(BaseModel):
    """Anthropic API configuration."""
    api_key: Optional[str] = Field(default=None)
    model: str = Field(default="claude-3-7-sonnet-20250219")
    max_tokens: int = Field(default=4000)
    temperature: float = Field(default=0.1)


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
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
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
        result[section] = {}
        for key, value in config[section].items():
            # Try to parse values
            try:
                # Handle boolean values
                if value.lower() in ("true", "yes", "1"):
                    result[section][key] = True
                elif value.lower() in ("false", "no", "0"):
                    result[section][key] = False
                # Handle numeric values
                elif value.isdigit():
                    result[section][key] = int(value)
                elif value.replace(".", "", 1).isdigit():
                    result[section][key] = float(value)
                # Handle lists
                elif "," in value:
                    result[section][key] = [v.strip() for v in value.split(",")]
                # Otherwise, keep as string
                else:
                    result[section][key] = value
            except Exception:
                # Fall back to string if parsing fails
                result[section][key] = value
                
    return result


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
    
    # Override with environment variables
    # Format: LLM_DOCS__SECTION__KEY
    for key, value in os.environ.items():
        if key.startswith("LLM_DOCS__"):
            parts = key.split("__")
            if len(parts) == 3:
                _, section, key = parts
                section = section.lower()
                key = key.lower()
                
                if section not in config_dict:
                    config_dict[section] = {}
                    
                # Parse the value
                if value.lower() in ("true", "yes", "1"):
                    config_dict[section][key] = True
                elif value.lower() in ("false", "no", "0"):
                    config_dict[section][key] = False
                elif value.isdigit():
                    config_dict[section][key] = int(value)
                elif value.replace(".", "", 1).isdigit():
                    config_dict[section][key] = float(value)
                else:
                    config_dict[section][key] = value
    
    # Special case for Anthropic API key
    if "ANTHROPIC_API_KEY" in os.environ and "anthropic" in config_dict:
        config_dict["anthropic"]["api_key"] = os.environ["ANTHROPIC_API_KEY"]
        
    # Create the config object
    return Config(**config_dict)


# Global config instance
config = load_config()
