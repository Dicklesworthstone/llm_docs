"""
Database connection and session management for llm_docs.
"""

import os
from typing import Generator
from pathlib import Path
from sqlmodel import SQLModel, Session, create_engine

# Default database path
DEFAULT_DB_PATH = Path("llm_docs.db")

# Get database URL from environment or use default SQLite file
DATABASE_URL = os.environ.get(
    "LLM_DOCS_DATABASE_URL", 
    f"sqlite:///{DEFAULT_DB_PATH}"
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL query logging
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,  # Check connection health before using
    pool_recycle=3600,  # Recycle connections hourly
)


def get_session() -> Generator[Session, None, None]:
    """
    Get a database session.
    
    Yields:
        Session: SQLModel session
    """
    with Session(engine) as session:
        yield session


def init_db() -> None:
    """Create all database tables."""
    from llm_docs.storage.models import create_db_and_tables
    create_db_and_tables(engine)


def reset_db() -> None:
    """Drop and recreate all database tables. Use with caution!"""
    SQLModel.metadata.drop_all(engine)
    
    from llm_docs.storage.models import create_db_and_tables
    create_db_and_tables(engine)
