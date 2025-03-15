"""
Tests for the documentation extraction module.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path
import tempfile

from llm_docs.doc_extraction import DocumentationExtractor
from llm_docs.storage.models import Package


@pytest.mark.asyncio
async def test_find_documentation_url():
    """Test finding documentation URL using Google search."""
    with patch('browser_use.Browser') as mock_browser:
        # Mock the browser response
        mock_browser_instance = AsyncMock()
        mock_page = AsyncMock()
        mock_browser_instance.__aenter__.return_value = mock_browser_instance
        mock_browser_instance.new_page.return_value = mock_page
        mock_page.evaluate.return_value = [
            "https://numpy.org/doc/stable/",
            "https://example.com/not-docs"
        ]
        mock_browser.return_value = mock_browser_instance
        
        # Create temp dir for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create extractor instance
            extractor = DocumentationExtractor(output_dir=temp_dir)
            
            # Call the method
            url = await extractor.find_documentation_url("numpy")
            
            # Check the result
            assert url == "https://numpy.org/doc/stable/"
            
            # Check that the browser was used correctly
            mock_browser_instance.new_page.assert_called_once()
            mock_page.goto.assert_called_once_with("https://www.google.com/search?q=numpy python documentation")
            mock_page.evaluate.assert_called_once()


@pytest.mark.asyncio
async def test_extract_content_from_page():
    """Test extracting content from a documentation page."""
    with patch('browser_use.Browser') as mock_browser:
        # Mock the browser response
        mock_browser_instance = AsyncMock()
        mock_page = AsyncMock()
        mock_browser_instance.__aenter__.return_value = mock_browser_instance
        mock_browser_instance.new_page.return_value = mock_page
        mock_page.content.return_value = """
        <html>
            <body>
                <nav>Navigation</nav>
                <main>
                    <h1>API Reference</h1>
                    <p>This is the API reference for the package.</p>
                </main>
                <footer>Footer</footer>
            </body>
        </html>
        """
        mock_browser.return_value = mock_browser_instance
        
        # Create temp dir for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create extractor instance
            extractor = DocumentationExtractor(output_dir=temp_dir)
            
            # Call the method
            content = await extractor.extract_content_from_page("https://example.com/docs")
            
            # Check the result - should contain the main content but not nav/footer
            assert "<h1>API Reference</h1>" in content
            assert "<p>This is the API reference for the package.</p>" in content
            assert "Navigation" not in content
            assert "Footer" not in content


@pytest.mark.asyncio
async def test_process_package_documentation():
    """Test processing package documentation."""
    with patch.object(DocumentationExtractor, 'find_documentation_url') as mock_find_url, \
         patch.object(DocumentationExtractor, 'map_documentation_site') as mock_map_site, \
         patch.object(DocumentationExtractor, 'extract_content_from_page') as mock_extract, \
         patch.object(DocumentationExtractor, 'convert_html_to_markdown') as mock_convert:
        
        # Mock the responses
        mock_find_url.return_value = "https://example.com/docs"
        mock_map_site.return_value = [
            "https://example.com/docs/index.html",
            "https://example.com/docs/api.html"
        ]
        mock_extract.side_effect = [
            "<h1>Introduction</h1><p>Welcome to the docs.</p>",
            "<h1>API</h1><p>API documentation here.</p>"
        ]
        mock_convert.side_effect = [
            "# Introduction\n\nWelcome to the docs.",
            "# API\n\nAPI documentation here."
        ]
        
        # Create temp dir for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create extractor instance
            extractor = DocumentationExtractor(output_dir=temp_dir)
            
            # Create a package
            package = Package(name="test-package")
            
            # Call the method
            doc_path = await extractor.process_package_documentation(package)
            
            # Check the result
            assert doc_path is not None
            assert Path(doc_path).exists()
            
            # Read the file and check its content
            with open(doc_path, 'r') as f:
                content = f.read()
                assert "# test-package Documentation" in content
                assert "# Introduction" in content
                assert "Welcome to the docs" in content
                assert "# API" in content
                assert "API documentation here" in content
