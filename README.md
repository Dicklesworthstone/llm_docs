# LLM Docs

A fully automated system for collecting and distilling Python package documentation into LLM-friendly formats, with support for multiple LLM providers.

## Executive Summary

LLM Docs automates the entire process of transforming standard Python package documentation into formats that Large Language Models can process more effectively. This leads to:

- **Better accuracy** when LLMs answer questions about Python libraries
- **Reduced token usage** by eliminating verbose, redundant content
- **Standardized knowledge representation** across different documentation styles
- **Scalable processing** of the entire Python package ecosystem
- **Provider flexibility** with support for Anthropic, OpenAI, Google, Mistral, and other LLM providers

## Overview

LLM Docs solves a critical problem: standard documentation is written for humans but isn't optimized for consumption by Large Language Models (LLMs). This project implements a complete, automated pipeline that:

1. Discovers the most popular Python packages
2. Automatically extracts their documentation from the web
3. Processes and distills this documentation into formats optimized for LLMs
4. Leverages multiple LLM providers through a unified interface

The entire system is designed to be generic and automated, requiring no package-specific implementations. It can process any Python package's documentation in a standardized way without manual intervention, and you can easily switch between different LLM providers for different parts of the system.

## Project Architecture

LLM Docs implements a two-stage pipeline:

### Stage 1: Documentation Collection

#### Package Discovery
- Automatically identifies the most popular Python packages based on download statistics
- Uses browser automation to scrape package rankings from PyPI stats sites
- Stores package metadata in a database with prioritization based on popularity
- Implements intelligent fallbacks if primary data sources are unavailable

#### Documentation Extraction
- Automatically locates the documentation site for each package
- Maps the structure of documentation sites to identify all relevant pages
- Uses browser automation to extract content from each page
- Converts HTML to Markdown while preserving essential information
- Combines all pages into a single comprehensive markdown file per package

### Stage 2: Documentation Distillation

- Takes the combined original documentation from Stage 1
- Uses specialized templates to guide LLMs in condensing documentation
- Processes documentation in manageable chunks to handle large documentation sets
- Applies different distillation strategies based on documentation type (API reference, tutorial, etc.)
- Produces concise, structured documentation optimized for LLM consumption
- Supports multiple LLM providers (Anthropic, OpenAI, Google, Mistral, etc.) through the aisuite library

## Why This Matters

Standard documentation presents several challenges for LLMs:

1. **Context Window Limitations**: Most package documentation exceeds LLM context windows
2. **Signal-to-Noise Ratio**: Documentation contains marketing language, redundant examples, and verbose explanations
3. **Inconsistent Structure**: Documentation formats vary widely across packages
4. **Human-Oriented Presentation**: Content is organized for human reading patterns, not machine comprehension

LLM Docs solves these problems by:

1. Systematically collecting comprehensive documentation
2. Condensing it to essential technical content
3. Structuring it for optimal LLM consumption
4. Preserving all critical information while eliminating noise
5. Providing flexibility to use the most suitable LLM provider for each task

The result: LLMs can provide more accurate, helpful responses about Python libraries while using fewer tokens.

## Command-Line Interface

In addition to the Python API, LLM Docs provides a convenient command-line interface:

```bash
# Discover packages and store in database
llm-docs discover --limit 100 --process 0

# Process (i.e. extract and distill) a specific package
llm-docs process numpy

# Only run discovery phase
llm-docs discover --top 1000 --save-db ./package_db.sqlite

```

## Installation

```bash
# Clone the repository
git clone https://github.com/Dicklesworthstone/llm_docs.git
cd llm_docs

# Create a virtual environment with UV (recommended)
# Install UV if needed: pip install uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# For development
uv pip install -r requirements-dev.txt

# Set up your .env file
cp .env.template .env
# Edit .env to add your API keys
```

### Dependencies

- Python 3.11+ (Note: This requirement is due to the `browser-use` dependency needing Python 3.11 or newer)
- `browser-use`: For web automation
- `markitdown`: For HTML to Markdown conversion
- `httpx`: For async HTTP requests
- `sqlmodel`: For database operations
- `aisuite`: For unified LLM provider interface
- `python-decouple`: For environment variable management
- `tqdm`: For progress indicators
- `rich`: For console output formatting

## Usage

### Basic Usage

```python
import asyncio
from llm_docs.storage.db import get_async_session
from llm_docs.discovery import PackageDiscovery
from llm_docs.doc_extraction import DocumentationExtractor
from llm_docs.distillation import DocumentationDistiller

async def main():
    # Initialize database session
    async with get_async_session() as session:
        # Stage 1A: Discover packages
        discovery = PackageDiscovery(session)
        await discovery.discover_and_store_packages(limit=100)
        
        # Get next batch of packages to process
        packages = await discovery.get_next_packages_to_process(limit=10)
        
        # Stage 1B: Extract documentation
        extractor = DocumentationExtractor(output_dir="original_docs")
        
        for package in packages:
            # Extract and combine documentation
            original_doc_path = await extractor.process_package_documentation(package)
            
            if original_doc_path:
                # Stage 2: Distill documentation
                distiller = DocumentationDistiller(
                    output_dir="distilled_docs",
                    # You can override the LLM provider configuration here
                    llm_config={
                        "provider": "openai",  # Use OpenAI instead of the default
                        "model": "gpt-4o",     # Specify which model to use
                        "temperature": 0.1,
                        "max_tokens": 4000
                    }
                )
                distilled_doc_path = await distiller.distill_documentation(
                    package,
                    original_doc_path
                )
                
                print(f"Distilled documentation saved to: {distilled_doc_path}")
                
        # Close clients
        await discovery.close()
        await extractor.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration

The system supports multiple configuration methods:

1. **Configuration File**: Edit `llm-docs.conf` to set system-wide defaults
2. **Environment Variables**: Set variables in your environment or in a `.env` file
3. **Programmatic Configuration**: Pass configuration directly when creating components

#### Configuration File Example

```ini
[database]
url = sqlite+aiosqlite:///llm_docs.db

[llm.default]
provider = anthropic
model = claude-3-7-sonnet-20250219
max_tokens = 4000
temperature = 0.1

[llm.distillation]
provider = openai
model = gpt-4o
max_tokens = 4000
temperature = 0.1
```

#### Environment File Example

```ini
# API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Configuration
LLM_DOCS__LLM__DEFAULT__PROVIDER=anthropic
LLM_DOCS__LLM__DEFAULT__MODEL=claude-3-7-sonnet-20250219
LLM_DOCS__LLM__DISTILLATION__PROVIDER=openai
LLM_DOCS__LLM__DISTILLATION__MODEL=gpt-4o
```

### LLM Provider Configuration

LLM Docs now supports multiple LLM providers through the aisuite library. You can configure different providers for different parts of the system:

- **Default Provider**: Used if no specific provider is configured for a component
- **Distillation Provider**: Used specifically for distilling documentation
- **Browser Exploration Provider**: Used for browser automation tasks
- **Documentation Extraction Provider**: Used for extracting and processing documentation

Supported providers include:
- Anthropic (Claude models)
- OpenAI (GPT models)
- Google (Gemini models)
- Mistral
- Azure OpenAI
- Groq
- Sambanova
- Watsonx
- Huggingface
- Ollama

To use a provider, you'll need to set the appropriate API key in your `.env` file or environment variables, and specify the provider and model in your configuration.

## System Components

### Package Discovery (`discovery.py`)

The discovery module is responsible for finding the most popular Python packages to process:

- Uses browser automation to extract package ranking information
- Implements multiple data source strategies with fallbacks
- Stores package metadata and download statistics in a database
- Prioritizes packages based on their popularity
- Manages a processing queue for systematic documentation extraction

### Documentation Extraction (`extractor.py`)

The extraction module handles all aspects of collecting and processing package documentation:

- Automatically locates documentation sites using multiple strategies
- Maps documentation site structure to find all relevant pages
- Extracts content while filtering out navigation, headers, footers, etc.
- Converts HTML to Markdown with proper formatting
- Combines multiple pages into a single comprehensive document

### Documentation Distillation (`distiller.py`)

The distillation module uses specialized templates to guide LLMs in condensing documentation:

- Provides specific instructions for different documentation types
- Handles documentation in manageable chunks to work within LLM context limits
- Ensures technical accuracy while removing verbosity and redundancy
- Maintains a consistent structure throughout the distilled output
- Supports multiple LLM providers through a unified interface

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This project is designed to be fully automated and generic, requiring no package-specific implementations. The entire pipeline from discovery to distillation is built to work with any Python package's documentation in a standardized way.