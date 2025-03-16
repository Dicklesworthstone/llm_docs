# LLM Docs

A fully automated system for collecting and distilling Python package documentation into LLM-friendly formats.

## Executive Summary

LLM Docs automates the entire process of transforming standard Python package documentation into formats that Large Language Models can process more effectively. This leads to:

- **Better accuracy** when LLMs answer questions about Python libraries
- **Reduced token usage** by eliminating verbose, redundant content
- **Standardized knowledge representation** across different documentation styles
- **Scalable processing** of the entire Python package ecosystem

## Overview

LLM Docs solves a critical problem: standard documentation is written for humans but isn't optimized for consumption by Large Language Models (LLMs). This project implements a complete, automated pipeline that:

1. Discovers the most popular Python packages
2. Automatically extracts their documentation from the web
3. Processes and distills this documentation into formats optimized for LLMs

The entire system is designed to be generic and automated, requiring no package-specific implementations. It can process any Python package's documentation in a standardized way without manual intervention.

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

The result: LLMs can provide more accurate, helpful responses about Python libraries while using fewer tokens.

## Command-Line Interface

In addition to the Python API, LLM Docs provides a convenient command-line interface:

```bash
# Process top 100 packages
llm-docs process --top 100 --output-dir ./docs

# Process specific packages
llm-docs process --packages numpy,pandas,requests --output-dir ./docs

# Only run discovery phase
llm-docs discover --top 1000 --save-db ./package_db.sqlite

# Only run extraction phase for packages in database
llm-docs extract --from-db ./package_db.sqlite --limit 10 --output-dir ./original_docs

# Only run distillation phase
llm-docs distill --input-dir ./original_docs --output-dir ./distilled_docs --type api
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
```

### Dependencies

- Python 3.8+
- `browser-use`: For web automation
- `markitdown`: For HTML to Markdown conversion
- `httpx`: For async HTTP requests
- `sqlmodel`: For database operations
- `tqdm`: For progress indicators
- `rich`: For console output formatting

## Usage

### Basic Usage

```python
import asyncio
from llm_docs.storage.db import get_async_session
from llm_docs.discovery import PackageDiscovery
from llm_docs.doc_extraction import DocumentationExtractor
from llm_docs.distillation import distill_documentation

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
                distilled_doc_path = distill_documentation(
                    package_name=package.name,
                    doc_path=original_doc_path,
                    output_dir="distilled_docs",
                    doc_type="api"  # or "tutorial", "quick_reference", etc.
                )
                
                print(f"Distilled documentation saved to: {distilled_doc_path}")
                
        # Close clients
        await discovery.close()
        await extractor.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration

The system is highly configurable:

```python
# Configure package discovery
discovery = PackageDiscovery(
    db_session=session,
    cache_dir="custom_cache_dir"  # Custom cache location
)

# Configure documentation extraction
extractor = DocumentationExtractor(
    output_dir="custom_output",  # Where to save original docs
    cache_dir="extraction_cache"  # Cache for extraction results
)

# Configure distillation
distilled_path = distill_documentation(
    package_name="package_name",
    doc_path="path/to/original_docs.md",
    output_dir="distilled_output",
    doc_type="api",  # Documentation type
    chunk_size=8000,  # Size of chunks for processing
    num_parts=5      # Number of parts to split into
)
```

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

### Documentation Distillation

The distillation module uses specialized templates to guide LLMs in condensing documentation:

- Provides specific instructions for different documentation types
- Handles documentation in manageable chunks to work within LLM context limits
- Ensures technical accuracy while removing verbosity and redundancy
- Maintains a consistent structure throughout the distilled output

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This project is designed to be fully automated and generic, requiring no package-specific implementations. The entire pipeline from discovery to distillation is built to work with any Python package's documentation in a standardized way.