"""Tests for chunking module."""
import pytest
from contextlab.chunking import chunk_text, split_on_headings, count_tokens


def test_split_on_headings():
    """Test heading-aware splitting."""
    text = """# Section 1
Some text here.

## Subsection 1.1
More text.

# Section 2
Final text."""
    
    sections = split_on_headings(text)
    assert len(sections) == 3
    assert sections[0][0] == "Section 1"
    assert sections[1][0] == "Subsection 1.1"
    assert sections[2][0] == "Section 2"


def test_no_empty_chunks():
    """Test that no empty chunks are produced."""
    from pathlib import Path
    from contextlab.chunking import chunk_file
    import tempfile
    
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Test\n\nSome content here.\n")
        temp_path = f.name
    
    chunks = chunk_file(temp_path)
    
    for chunk in chunks:
        assert chunk.text.strip(), f"Empty chunk found: {chunk.chunk_id}"
        assert chunk.token_count > 0, f"Zero-token chunk: {chunk.chunk_id}"


def test_overlap_exists():
    """Test that overlapping chunks are produced for long sections."""
    # Create a long text that exceeds chunk size
    text = "word " * 500  # 500 words
    
    chunks = chunk_text(
        text, 
        doc_id="test", 
        doc_path="/test", 
        chunk_tokens=100, 
        chunk_overlap_tokens=20
    )
    
    assert len(chunks) > 1, "Long text should produce multiple chunks"
    
    # Check overlap: last words of chunk N should appear in chunk N+1
    for i in range(len(chunks) - 1):
        chunk1_text = chunks[i].text
        chunk2_text = chunks[i + 1].text
        
        # Extract last 5 tokens from chunk1 and first 5 from chunk2
        words1 = chunk1_text.split()
        words2 = chunk2_text.split()
        
        # There should be some overlap
        overlap = set(words1[-10:]) & set(words2[:10])
        assert len(overlap) > 0, f"No overlap between chunks {i} and {i+1}"


def test_stable_chunk_ids():
    """Test that same file produces same chunk IDs on re-chunk."""
    from pathlib import Path
    from contextlab.chunking import chunk_file
    import tempfile
    import os
    
    text = "# Test Document\n\nSome content for testing.\n\n## Section 2\n\nMore content here.\n"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(text)
        temp_path = f.name
    
    chunks1 = chunk_file(temp_path)
    chunk_ids1 = [c.chunk_id for c in chunks1]
    
    chunks2 = chunk_file(temp_path)
    chunk_ids2 = [c.chunk_id for c in chunks2]
    
    assert chunk_ids1 == chunk_ids2, "Chunk IDs should be stable across re-chunking"
    
    os.unlink(temp_path)


def test_token_count():
    """Test token counting."""
    text = "Hello world this is a test."
    count = count_tokens(text)
    assert count > 0, "Should count some tokens"


def test_chunk_id_format():
    """Test chunk_id format matches spec."""
    from pathlib import Path
    from contextlab.chunking import chunk_file
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Test\n\nContent here.\n")
        temp_path = f.name
    
    chunks = chunk_file(temp_path)
    
    import os
    os.unlink(temp_path)
    
    for chunk in chunks:
        # Format: {doc_id}::c{n:04d}
        assert "::c" in chunk.chunk_id, f"Invalid chunk_id format: {chunk.chunk_id}"
        assert chunk.doc_id in chunk.chunk_id, "doc_id should be in chunk_id"
