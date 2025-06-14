"""
Tests for the documentation distillation module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_docs.distillation import DocumentationDistiller
from llm_docs.storage.models import Package


class LLMAPIError(Exception):
    """Exception raised when LLM API calls fail after retries."""
    pass

@pytest.mark.asyncio
async def test_split_into_chunks():
    """Test splitting markdown content into chunks."""
    # Create temp dir for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create distiller instance
        distiller = DocumentationDistiller(output_dir=temp_dir)
        
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
        all_chunks = "".join(chunks)
        for header in ["Section 1", "Section 2", "Section 3", "Section 4", "Section 5"]:
            assert header in all_chunks


@pytest.mark.asyncio
async def test_distill_chunk():
    """Test distilling a chunk of documentation."""
    with patch('llm_docs.distillation.distiller.ai.Client') as MockAiClient:
        # Configure the mock client instance and its nested methods
        mock_client_instance = MockAiClient.return_value
        mock_create = mock_client_instance.chat.completions.create
        
        # Mock the aisuite response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Distilled content"))]
        mock_create.return_value = mock_response
        
        # Create temp dir for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create distiller instance
            distiller = DocumentationDistiller(output_dir=temp_dir)
            
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
            distiller = DocumentationDistiller(output_dir=temp_dir)
            
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

@pytest.mark.asyncio
async def test_distill_chunk_retry_on_error():
    """Test retrying when API call fails."""
    with patch('llm_docs.distillation.distiller.ai.Client') as MockAiClient:
        # Configure the mock client instance and its nested methods
        mock_client_instance = MockAiClient.return_value
        mock_create = mock_client_instance.chat.completions.create

        # Simulate API error on first call, success on second
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Distilled content"))]
        mock_create.side_effect = [
            Exception("API Error"),
            mock_response
        ]
        
        # Create distiller with short retry delay
        with tempfile.TemporaryDirectory() as temp_dir:
            distiller = DocumentationDistiller(output_dir=temp_dir, retry_delay=0.1)
            
            # Call the method
            result = await distiller.distill_chunk(
                package_name="test-package",
                chunk_content="Original content",
                part_num=1,
                num_parts=3
            )
            
            # Check the result
            assert result == "Distilled content"
            
            # Check that the API was called twice
            assert mock_create.call_count == 2

@pytest.mark.asyncio
async def test_distill_chunk_all_retries_fail():
    """Test behavior when all API retry attempts fail."""
    with patch('llm_docs.distillation.distiller.ai.Client') as MockAiClient, \
         tempfile.TemporaryDirectory() as temp_dir:
        
        # Configure the mock client instance and its nested methods
        mock_client_instance = MockAiClient.return_value
        mock_create = mock_client_instance.chat.completions.create

        # Mock API to always fail
        mock_create.side_effect = Exception("API Error")
        
        # Create distiller with minimal retry settings for test
        distiller = DocumentationDistiller(
            output_dir=temp_dir,
            retry_delay=0.1,
            max_retries=2
        )
        
        # Verify it raises the expected exception
        with pytest.raises(LLMAPIError) as excinfo:
            await distiller.distill_chunk(
                package_name="test-package",
                chunk_content="Test content",
                part_num=1,
                num_parts=3
            )
        
        # Check the error message
        assert "Failed to distill chunk" in str(excinfo.value)
        assert "API Error" in str(excinfo.value)
        
        # Verify the mock was called the expected number of times (equal to max_retries)
        assert mock_create.call_count == 2            