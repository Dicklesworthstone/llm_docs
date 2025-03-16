"""
Tests for the package discovery module.
"""

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.future import select

from llm_docs.package_discovery import PackageDiscovery
from llm_docs.storage.models import Package, PackageStats


@pytest.mark.asyncio
async def test_get_top_packages():
    """Test getting top packages from PyPI."""
    with patch('browser_use.Browser') as mock_browser:
        # Mock the browser response
        mock_browser_instance = AsyncMock()
        mock_page = AsyncMock()
        mock_browser_instance.__aenter__.return_value = mock_browser_instance
        mock_browser_instance.new_page.return_value = mock_page
        mock_page.evaluate.return_value = ["numpy", "pandas", "requests"]
        mock_browser.return_value = mock_browser_instance
        
        # Mock session
        mock_session = AsyncMock()
        
        # Create discovery instance
        discovery = PackageDiscovery(mock_session)
        
        # Call the method
        packages = await discovery.get_top_packages(limit=3)
        
        # Check the result
        assert packages == ["numpy", "pandas", "requests"]
        
        # Check that the browser was used correctly
        mock_browser_instance.new_page.assert_called_once()
        mock_page.goto.assert_called_once_with("https://pypistats.org/top-packages")
        mock_page.evaluate.assert_called_once()


@pytest.mark.asyncio
async def test_get_package_stats():
    """Test getting package stats from PyPI."""
    with patch('httpx.AsyncClient.get') as mock_get:
        # Mock the response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "last_day": 100000,
                "last_week": 700000,
                "last_month": 3000000
            }
        }
        mock_get.return_value = mock_response
        
        # Mock session
        mock_session = AsyncMock()
        
        # Create discovery instance
        discovery = PackageDiscovery(mock_session)
        
        # Call the method
        stats = await discovery.get_package_stats("numpy")
        
        # Check the result
        assert stats["data"]["last_month"] == 3000000
        
        # Check that the API was called correctly
        mock_get.assert_called_once_with("https://pypistats.org/api/packages/numpy/recent")


@pytest.mark.asyncio
async def test_discover_and_store_packages(session):
    """Test discovering and storing packages."""
    with patch.object(PackageDiscovery, 'get_top_packages') as mock_get_top, \
         patch.object(PackageDiscovery, 'get_package_stats') as mock_get_stats:
        
        # Mock the responses
        mock_get_top.return_value = ["numpy", "pandas"]
        
        mock_get_stats.side_effect = [
            {"data": {"last_month": 3000000}},
            {"data": {"last_month": 2000000}}
        ]
        
        # Create discovery instance
        discovery = PackageDiscovery(session)
        
        # Call the method
        packages = await discovery.discover_and_store_packages(limit=2)
        
        # Check the result
        assert len(packages) == 2
        assert packages[0].name == "numpy"
        assert packages[1].name == "pandas"
        
        # Check that the database was updated
        result = await session.execute(select(Package))
        db_packages = result.scalars().all()
        assert len(db_packages) == 2

        result = await session.execute(select(PackageStats))
        db_stats = result.scalars().all()
        assert len(db_stats) == 2
