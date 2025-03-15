"""
Tests for the documentation distillation module.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import tempfile
import os
from pathlib import Path

from llm_docs.distillation import DocumentationDistiller
from llm_docs.storage.models import Package


@pytest.mark.asyncio
async def test_split_into_chunks():
    """Test splitting markdown content into chunks."""
    # Create temp dir for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create distiller instance
        distiller = DocumentationDistiller(output_dir=temp_dir, api_key="fake-key")
        
        # Create test content with sections
        content = """
# Section 1
This is the content of section 1.

## Subsection 1.1
More content here.

# Section 2
This is the content of section 2.

# Section 3
This is the content of section 3.

# Section 4
This is the content of section 4.

# Section 5
This is the content of section 5.
"""
        
        # Split into chunks
        chunks = distiller.split_into_chunks(content, num_chunks=3)
        
        # Check results
        assert len(chunks) == 3
        assert "Section 1" in chunks[0]
        assert "Section 2" in chunks[1]
        assert "Section 3" in chunks[0]
        assert "Section 4" in chunks[1]
        assert "Section 5" in chunks[2]


@pytest.mark.asyncio
async def test_distill_chunk():
    """Test distilling a chunk of documentation."""
    with patch('anthropic.AsyncAnthropic.messages.create') as mock_create:
        # Mock the Anthropic API response
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Distilled content")]
        mock_create.return_value = mock_message
        
        # Create temp dir for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create distiller instance
            distiller = DocumentationDistiller(output_dir=temp_dir, api_key="fake-key")
            
            # Call the method
            result = await distiller.distill_chunk(
                package_name="test-package",
                chunk_content="Original content",
                part_num=1,
                num_parts=3,
                package_description="a test package"
            )
            
            # Check the result
            assert result == "Distilled content"
            
            # Check that the API was called correctly
            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            assert kwargs["model"] == "claude-3-7-sonnet-20250219"
            assert kwargs["max_tokens"] == 4000
            assert kwargs["temperature"] == 0.1
            assert len(kwargs["messages"]) == 1
            assert kwargs["messages"][0]["role"] == "user"
            assert "test-package" in kwargs["messages"][0]["content"]
            assert "Original content" in kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_distill_documentation():
    """Test distilling full documentation."""
    with patch.object(DocumentationDistiller, 'split_into_chunks') as mock_split, \
         patch.object(DocumentationDistiller, 'distill_chunk') as mock_distill:
        
        # Mock the responses
        mock_split.return_value = ["Chunk 1", "Chunk 2", "Chunk 3"]
        mock_distill.side_effect = [
            "Distilled chunk 1",
            "Distilled chunk 2",
            "Distilled chunk 3",
            "Additional content"
        ]
        
        # Create temp dir and test file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test markdown file
            doc_path = os.path.join(temp_dir, "test_original.md")
            with open(doc_path, 'w') as f:
                f.write("# Test Documentation\n\nThis is test content.")
            
            # Create distiller instance
            distiller = DocumentationDistiller(output_dir=temp_dir, api_key="fake-key")
            
            # Create a package
            package = Package(name="test-package")
            
            # Call the method
            distilled_path = await distiller.distill_documentation(package, doc_path)
            
            # Check the result
            assert distilled_path is not None
            assert Path(distilled_path).exists()
            
            # Read the file and check its content
            with open(distilled_path, 'r') as f:
                content = f.read()
                assert "test-package - Distilled LLM-Optimized Documentation" in content
                assert "Distilled chunk 1" in content
                assert "Distilled chunk 2" in content
                assert "Distilled chunk 3" in content
                assert "Additional content" in content
