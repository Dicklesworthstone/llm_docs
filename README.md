# llm_docs

## Overview

llm_docs provides condensed, optimized documentation specifically tailored for efficient consumption by Large Language Models (LLMs). Traditional documentation, primarily designed for humans, often includes redundant content, extraneous formatting, and promotional material that unnecessarily occupies valuable context window space in LLMs. This project aims to create streamlined documentation, maximizing the efficiency and effectiveness of LLM-assisted programming tasks.

## Installation

```bash
# Clone the repository
git clone https://github.com/Dicklesworthstone/llm_docs.git
cd llm_docs

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Usage

### Command Line Interface

```bash
# Discover top packages
llm_docs discover --limit 100

# Process a specific package
llm_docs process numpy

# Start the API server
llm_docs serve
```

### API

The API server provides endpoints for managing and accessing distilled documentation.

```bash
# Start the API server
uvicorn llm_docs.api.app:app --reload
```

## License

MIT License
