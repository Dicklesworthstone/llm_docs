"""
Pytest configuration for llm_docs tests.
"""


import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine


# Use in-memory SQLite for tests
@pytest.fixture(name="engine")
def engine_fixture():
    """Create a SQLite in-memory engine for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    yield engine
    SQLModel.metadata.drop_all(engine)


@pytest.fixture(name="session")
def session_fixture(engine):
    """Create a database session for testing."""
    with Session(engine) as session:
        yield session


@pytest.fixture(name="client")
def client_fixture(engine):
    """Create a FastAPI test client."""
    # We need to modify the database dependency
    # Import here to avoid circular imports
    from llm_docs.api.app import app
    from llm_docs.storage.database import get_session
    
    def get_test_session():
        with Session(engine) as session:
            yield session
    
    # Override the dependency
    app.dependency_overrides[get_session] = get_test_session
    
    # Create the client
    with TestClient(app) as client:
        yield client
    
    # Clear the override
    app.dependency_overrides.clear()
