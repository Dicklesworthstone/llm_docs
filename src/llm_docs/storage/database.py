"""
Database connection and session management for llm_docs.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from rich.console import Console
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel, text

# Console for colored output
console = Console()

# Default database path
DEFAULT_DB_PATH = Path("llm_docs.db")

# Get database URL from environment or use default SQLite file
DATABASE_URL = os.environ.get(
    "LLM_DOCS_DATABASE_URL", 
    f"sqlite+aiosqlite:///{DEFAULT_DB_PATH}"
)

# Create engine with proper async support
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL query logging
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    poolclass=NullPool  # Avoid connection pooling issues with SQLite
)

# Create async session maker
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False  # Important for async operations
)

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async session for database operations.
    
    This is a context manager that will automatically close the session
    when the context is exited.
    
    Yields:
        AsyncSession: SQLModel async session
    """
    session = async_session_maker()
    try:
        yield session
    finally:
        await session.close()

async def set_sqlite_pragmas():
    """Set SQLite pragmas for better performance"""
    if "sqlite" in DATABASE_URL:
        async with engine.begin() as conn:
            await conn.execute(text("PRAGMA journal_mode=WAL;"))
            await conn.execute(text("PRAGMA synchronous=NORMAL;"))
            await conn.execute(text("PRAGMA cache_size=-262144;"))
            await conn.execute(text("PRAGMA busy_timeout=10000;"))
            await conn.execute(text("PRAGMA wal_autocheckpoint=1000;"))
            await conn.execute(text("PRAGMA mmap_size=30000000000;"))
            await conn.execute(text("PRAGMA threads=4;"))
            await conn.execute(text("PRAGMA optimize;"))
            await conn.execute(text("PRAGMA secure_delete=OFF;"))
            await conn.execute(text("PRAGMA temp_store=MEMORY;"))
            await conn.execute(text("PRAGMA page_size=4096;"))
            await conn.execute(text("PRAGMA auto_vacuum=INCREMENTAL;"))
            await conn.execute(text("PRAGMA locking_mode=NORMAL;"))
            await conn.execute(text("PRAGMA foreign_keys=ON;"))
            console.print("[green]SQLite pragmas set for optimal performance[/green]")

async def init_db() -> None:
    """Create all database tables."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        await set_sqlite_pragmas()
        console.print("[green]Database initialized successfully[/green]")
    except Exception as e:
        console.print(f"[bold red]Error initializing database: {e}[/bold red]")
        raise

async def reset_db() -> None:
    """Drop and recreate all database tables. Use with caution!"""
    try:
        console.print("[yellow]Dropping all tables...[/yellow]")
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.drop_all)
        
        console.print("[yellow]Creating tables...[/yellow]")
        await init_db()
        console.print("[green]Database reset successfully[/green]")
    except Exception as e:
        console.print(f"[bold red]Error resetting database: {e}[/bold red]")
        raise

@asynccontextmanager
async def transaction():
    """
    Context manager for database transactions.
    Automatically rolls back if an exception occurs.
    """
    async with transaction() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise    