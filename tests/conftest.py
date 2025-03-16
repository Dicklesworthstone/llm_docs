"""
Pytest configuration for llm_docs tests.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlmodel import SQLModel


@pytest.fixture(scope="session", name="async_engine")
def async_engine_fixture():
    """Create an async SQLite in-memory engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False}
    )
    async def create_tables():
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
    asyncio.run(create_tables())
    yield engine
    async def drop_tables():
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.drop_all)
    asyncio.run(drop_tables())

@pytest.fixture(name="async_session")
async def async_session_fixture(async_engine):
    """Create an async database session for testing."""
    async with AsyncSession(async_engine) as session:
        yield session
        # Roll back all changes after the test
        await session.rollback()


@pytest.fixture(name="client")
def client_fixture(async_engine):
    """Create a FastAPI test client."""
    # Import here to avoid circular imports
    from llm_docs.api.app import app
    from llm_docs.storage.database import get_async_session
    
    async def get_test_session():
        async with AsyncSession(async_engine) as session:
            yield session

    # Override the dependency
    app.dependency_overrides[get_async_session] = get_test_session
    
    # Create the client
    with TestClient(app) as client:
        yield client
    
    # Clear the override
    app.dependency_overrides.clear()