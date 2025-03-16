"""
Tests for the FastAPI application.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from llm_docs.storage.models import DistillationJob, DistillationJobStatus, Package, PackageStatus


@pytest.mark.asyncio
async def test_get_stats(client: TestClient, async_session: AsyncSession):
    """Test getting system statistics."""
    # Add some test packages
    packages = [
        Package(
            name="package1",
            status=PackageStatus.DISTILLATION_COMPLETED,
            priority=100,
            original_doc_path="/tmp/doc1.md",
            distilled_doc_path="/tmp/distilled1.md"
        ),
        Package(
            name="package2",
            status=PackageStatus.DOCS_EXTRACTED,
            priority=90,
            original_doc_path="/tmp/doc2.md"
        ),
        Package(
            name="package3",
            status=PackageStatus.DISCOVERED,
            priority=80
        )
    ]
    
    for package in packages:
        async_session.add(package)
    
    await async_session.commit()
    
    # Make the request
    response = client.get("/")
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert data["total_packages"] == 3
    assert data["packages_with_docs"] == 2
    assert data["packages_distilled"] == 1
    assert len(data["popular_packages"]) == 3
    assert len(data["recent_distillations"]) == 1


@pytest.mark.asyncio
async def test_list_packages(client: TestClient, async_session: AsyncSession):
    """Test listing packages."""
    # Add some test packages
    packages = [
        Package(
            name="package1",
            status=PackageStatus.DISTILLATION_COMPLETED,
            priority=100
        ),
        Package(
            name="package2",
            status=PackageStatus.DOCS_EXTRACTED,
            priority=90
        ),
        Package(
            name="package3",
            status=PackageStatus.DISCOVERED,
            priority=80
        )
    ]
    
    for package in packages:
        async_session.add(package)
    
    await async_session.commit()
    
    # Make the request
    response = client.get("/packages")
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    
    # Check order (by priority)
    assert data[0]["name"] == "package1"
    assert data[1]["name"] == "package2"
    assert data[2]["name"] == "package3"
    
    # Test filtering by status
    response = client.get("/packages?status=discovered")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "package3"


@pytest.mark.asyncio
async def test_get_package(client: TestClient, async_session: AsyncSession):
    """Test getting a specific package."""
    # Add a test package
    package = Package(
        name="test-package",
        status=PackageStatus.DISTILLATION_COMPLETED,
        priority=100,
        original_doc_path="/tmp/doc.md",
        distilled_doc_path="/tmp/distilled.md"
    )
    
    async_session.add(package)
    await async_session.commit()
    await async_session.refresh(package)
    
    # Make the request
    response = client.get(f"/packages/{package.id}")
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test-package"
    assert data["status"] == "distillation_completed"
    assert data["original_doc_path"] == "/tmp/doc.md"
    assert data["distilled_doc_path"] == "/tmp/distilled.md"
    
    # Test non-existent package
    response = client.get("/packages/9999")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_package(client: TestClient, async_session: AsyncSession):
    """Test creating a new package."""
    # Make the request
    response = client.post(
        "/packages",
        json={"name": "new-package", "priority": 50, "description": "A test package"}
    )
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "new-package"
    assert data["status"] == "discovered"
    assert data["priority"] == 50
    assert data["description"] == "A test package"
    
    # Check that the package was created in the database
    result = await async_session.execute(
        select(Package).where(Package.name == "new-package")
    )
    package = result.scalar_one_or_none()
    assert package is not None
    assert package.name == "new-package"
    assert package.priority == 50
    
    # Test creating a duplicate package
    response = client.post(
        "/packages",
        json={"name": "new-package"}
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_list_jobs(client: TestClient, async_session: AsyncSession):
    """Test listing distillation jobs."""
    # Add a test package
    package = Package(
        name="test-package",
        status=PackageStatus.DISTILLATION_COMPLETED
    )
    
    async_session.add(package)
    await async_session.commit()
    await async_session.refresh(package)
    
    # Add some test jobs
    jobs = [
        DistillationJob(
            package_id=package.id,
            status=DistillationJobStatus.COMPLETED,
            input_file_path="/tmp/doc.md",
            output_file_path="/tmp/distilled.md"
        ),
        DistillationJob(
            package_id=package.id,
            status=DistillationJobStatus.FAILED,
            input_file_path="/tmp/doc2.md",
            error_message="Test error"
        )
    ]
    
    for job in jobs:
        async_session.add(job)
    
    await async_session.commit()
    
    # Make the request
    response = client.get("/jobs")
    
    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    
    # Test filtering by status
    response = client.get("/jobs?status=completed")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["status"] == "completed"
    assert data[0]["package_name"] == "test-package"

@pytest.mark.asyncio
async def test_create_package_invalid_name(client: TestClient):
    """Test creating a package with an invalid name."""
    # Make the request with an invalid package name
    response = client.post(
        "/packages",
        json={"name": "invalid name with spaces", "priority": 50}
    )
    
    # Check that it returns a validation error
    assert response.status_code == 400    

@pytest.mark.asyncio
async def test_concurrent_package_processing(client: TestClient, async_session: AsyncSession, mocker):
    """Test that concurrent requests to process the same package are handled correctly."""
    # Mock the actual processing function to avoid real processing
    mocker.patch(
        "llm_docs.api.app.process_package",
        side_effect=lambda pkg_id: asyncio.sleep(0.1)  # Just a small delay to simulate work
    )
    
    # Mock Redis for locking - first call succeeds, others fail
    mock_redis = mocker.patch("llm_docs.api.app.redis_client")
    mock_redis.set.side_effect = [True, False, False]  # First call gets lock, others don't
    
    # Create a test package
    package = Package(
        name="concurrent-test-package",
        status=PackageStatus.DISCOVERED
    )
    
    async_session.add(package)
    await async_session.commit()
    await async_session.refresh(package)
    
    # Make concurrent requests
    response1 = client.post(f"/packages/{package.id}/process")
    response2 = client.post(f"/packages/{package.id}/process")
    response3 = client.post(f"/packages/{package.id}/process")
    
    # First request should succeed, others should indicate the package is already being processed
    assert response1.status_code == 200
    assert "Processing started" in response1.json()["message"]
    
    assert response2.status_code == 400
    assert "already being processed" in response2.json()["detail"]
    
    assert response3.status_code == 400
    assert "already being processed" in response3.json()["detail"]
    
    # Verify that the mock was called correctly
    mock_redis.set.assert_called_with(
        f"processing:package:{package.id}", "1", nx=True, ex=3600
    )
    assert mock_redis.set.call_count == 3    

@pytest.mark.asyncio
async def test_discover_packages(client: TestClient, async_session: AsyncSession, mocker):
    """Test the discover packages endpoint."""
    # Create mock packages for the discovery result
    mock_packages = [
        Package(id=1, name="package1", priority=100),
        Package(id=2, name="package2", priority=90),
        Package(id=3, name="package3", priority=80)
    ]
    
    # Mock the PackageDiscovery.discover_and_store_packages method
    mock_discover = mocker.patch("llm_docs.package_discovery.PackageDiscovery.discover_and_store_packages")
    mock_discover.return_value = mock_packages
    
    # Mock the PackageDiscovery.close method
    mocker.patch("llm_docs.package_discovery.PackageDiscovery.close")
    
    # Make the request
    response = client.post("/discover?limit=10")
    
    # Check the response
    assert response.status_code == 200
    assert "Discovered 3 packages" in response.json()["message"]
    
    # Verify the mock was called correctly
    mock_discover.assert_called_once_with(10)
    
    # Test with process_top parameter
    mocker.patch("llm_docs.api.app.process_package")
    
    response = client.post("/discover?limit=10&process_top=2")
    
    assert response.status_code == 200
    assert "Started processing 2 packages" in response.json()["processing"]    