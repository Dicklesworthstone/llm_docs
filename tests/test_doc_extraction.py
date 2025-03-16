"""
Tests for the documentation extraction module.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

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

@pytest.mark.asyncio
async def test_extract_content_from_complex_page():
    """Test extracting content from a page with complex structure."""
    with patch('browser_use.Browser') as mock_browser:
        # Mock the browser response with a complex HTML structure
        mock_browser_instance = AsyncMock()
        mock_page = AsyncMock()
        mock_browser_instance.__aenter__.return_value = mock_browser_instance
        mock_browser_instance.new_page.return_value = mock_page
        
        # Create a complex HTML structure with nested elements, iframes, scripts, etc.
        complex_html = """
        <html>
            <head>
                <title>Complex Documentation</title>
                <script>var x = 1;</script>
                <style>.sidebar { display: none; }</style>
            </head>
            <body>
                <header>
                    <nav>
                        <ul>
                            <li><a href="/">Home</a></li>
                            <li><a href="/docs">Docs</a></li>
                        </ul>
                    </nav>
                </header>
                <div class="container">
                    <aside class="sidebar">
                        <ul>
                            <li><a href="#section1">Section 1</a></li>
                            <li><a href="#section2">Section 2</a></li>
                        </ul>
                    </aside>
                    <main class="content">
                        <h1>API Documentation</h1>
                        <section id="section1">
                            <h2>Function Documentation</h2>
                            <p>This is the documentation for function X.</p>
                            <pre><code>def x(): pass</code></pre>
                        </section>
                        <section id="section2">
                            <h2>Class Documentation</h2>
                            <p>This is the documentation for class Y.</p>
                            <div class="example">
                                <h3>Example</h3>
                                <pre><code>y = Y()</code></pre>
                            </div>
                        </section>
                    </main>
                </div>
                <footer>
                    <p>Copyright 2023</p>
                </footer>
                <script>
                    // Some JavaScript
                    function init() { console.log('initialized'); }
                </script>
            </body>
        </html>
        """
        
        mock_page.content.return_value = complex_html
        mock_browser.return_value = mock_browser_instance
        
        # Create extractor instance
        extractor = DocumentationExtractor()
        
        # Call the method
        result = await extractor.extract_content_from_page("https://example.com/docs/complex")
        
        # Check that the main content was extracted
        assert "<h1>API Documentation</h1>" in result
        assert "Function Documentation" in result
        assert "Class Documentation" in result
        assert "This is the documentation for function X." in result
        assert "This is the documentation for class Y." in result
        
        # Check that nav, header, footer, scripts were removed
        assert "<nav>" not in result
        assert "<header>" not in result
        assert "<footer>" not in result
        assert "<script>" not in result
        assert "Copyright 2023" not in result
        
        # Verify browser method calls
        mock_browser_instance.new_page.assert_called_once()
        mock_page.goto.assert_called_once_with("https://example.com/docs/complex", timeout=30000)
        mock_page.content.assert_called_once()

@pytest.mark.asyncio
async def test_extraction_error_handling():
    """Test how extraction handles errors."""
    # Test browser error
    with patch('browser_use.Browser') as mock_browser:
        # Mock browser to raise exception
        mock_browser_instance = AsyncMock()
        mock_browser_instance.__aenter__.side_effect = Exception("Browser error")
        mock_browser.return_value = mock_browser_instance
        
        extractor = DocumentationExtractor()
        
        # Should handle the error gracefully
        result = await extractor.extract_content_from_page("https://example.com")
        assert "error" in result.lower()
        assert "Browser error" in result
    
    # Test page loading error
    with patch('browser_use.Browser') as mock_browser:
        # Mock page.goto to raise exception
        mock_browser_instance = AsyncMock()
        mock_page = AsyncMock()
        mock_browser_instance.__aenter__.return_value = mock_browser_instance
        mock_browser_instance.new_page.return_value = mock_page
        mock_page.goto.side_effect = Exception("Page load error")
        mock_browser.return_value = mock_browser_instance
        
        extractor = DocumentationExtractor()
        
        # Should handle the error gracefully
        result = await extractor.extract_content_from_page("https://example.com")
        assert "error accessing" in result.lower()
        assert "Page load error" in result        