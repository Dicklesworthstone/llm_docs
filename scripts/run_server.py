#!/usr/bin/env python
"""
Script to run the llm_docs API server.
"""

import argparse
import os
import sys
from pathlib import Path

import uvicorn

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_docs.config import config
from llm_docs.storage.database import init_db


def main():
    """Run the server."""
    parser = argparse.ArgumentParser(description="Run the llm_docs API server")
    parser.add_argument(
        "--host",
        type=str,
        default=config.api.host,
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.api.port,
        help="Port to bind to"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload"
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize the database if it doesn't exist"
    )
    
    args = parser.parse_args()
    
    # Initialize database if needed
    if args.init_db or not os.path.exists("llm_docs.db"):
        print("Initializing database...")
        init_db()
    
    # Start the server
    print(f"Starting API server at http://{args.host}:{args.port}")
    uvicorn.run(
        "llm_docs.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
