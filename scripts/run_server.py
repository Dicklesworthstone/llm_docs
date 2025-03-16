#!/usr/bin/env python
"""
Script to run the llm_docs API server.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import uvicorn
from rich.console import Console

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_docs.config import config
from llm_docs.storage.database import init_db

# Initialize console
console = Console()

def init_db_sync():
    """Run the async init_db function in a new event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(init_db())
    finally:
        loop.close()

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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--no-access-log",
        action="store_true",
        help="Disable access logging"
    )
    
    args = parser.parse_args()
    
    # Initialize database if needed
    if args.init_db or not os.path.exists("llm_docs.db"):
        console.print("[yellow]Initializing database...[/yellow]")
        init_db_sync()  # Run async function in sync context
    
    # Start the server
    console.print(f"[green]Starting API server at http://{args.host}:{args.port}[/green]")
    console.print(f"[blue]Workers: {args.workers}, Reload: {args.reload}[/blue]")
    console.print("Press Ctrl+C to stop.")
    
    # Configure logging
    log_config = None
    if args.no_access_log:
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s [%(process)d] [%(levelname)s] %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr"
                }
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": "INFO"},
                "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
                "uvicorn.access": {"handlers": ["default"], "level": "ERROR", "propagate": False},
            }
        }
    
    # Start Uvicorn with the configured settings
    uvicorn.run(
        "llm_docs.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,  # Cannot use multiple workers with reload
        log_config=log_config
    )


if __name__ == "__main__":
    main()
