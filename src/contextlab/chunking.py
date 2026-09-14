"""Heading-aware chunking with provenance tracking."""
import re
import hashlib
from pathlib import Path

import tiktoken
from typing import Iterator

from contextlab.types import Chunk


def get_tokenizer(encoding: str = "cl100k_base") -> tiktoken.Encoding:
    """Get tiktoken encoder."""
    return tiktoken.get_encoding(encoding)


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken."""
    encoder = get_tokenizer(encoding)
    return len(encoder.encode(text))


def split_on_headings(text: str) -> list[tuple[str | None, str]]:
    """Split text on markdown headings, yielding (heading, body)."""
    lines = text.split("\n")
    chunks: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")
    
    for line in lines:
        m = heading_pattern.match(line)
        if m:
            # Yield previous section
            if current_lines:
                chunks.append((current_heading, "\n".join(current_lines)))
            current_heading = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)
    
    # Yield last section
    if current_lines:
        chunks.append((current_heading, "\n".join(current_lines)))
    
    return chunks


def chunk_text(
    text: str,
    doc_id: str,
    doc_path: str,
    chunk_tokens: int = 256,
    chunk_overlap_tokens: int = 32,
    encoding: str = "cl100k_base",
) -> list[Chunk]:
    """Split text into token-bounded chunks with heading awareness and provenance."""
    encoder = get_tokenizer(encoding)
    
    sections = split_on_headings(text)
    chunks: list[Chunk] = []
    chunk_index = 0
    
    for section_heading, section_text in sections:
        if not section_text.strip():
            continue
        
        # Tokenize the full section
        section_tokens = encoder.encode(section_text)
        total_tokens = len(section_tokens)
        
        if total_tokens == 0:
            continue
        
        # If section fits in one chunk, emit it
        if total_tokens <= chunk_tokens:
            chunk_text_content = encoder.decode(section_tokens)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::c{chunk_index:04d}",
                doc_id=doc_id,
                doc_path=doc_path,
                text=chunk_text_content,
                token_count=total_tokens,
                start_char=0,  # Relative to section start
                end_char=len(chunk_text_content),
                section=section_heading,
                metadata={},
            ))
            chunk_index += 1
            continue
        
        # Otherwise, slide window with overlap
        stride = chunk_tokens - chunk_overlap_tokens
        start = 0
        
        while start < total_tokens:
            end = min(start + chunk_tokens, total_tokens)
            chunk_tokens_list = section_tokens[start:end]
            
            # Decode back to text
            chunk_text_content = encoder.decode(chunk_tokens_list)
            
            # Calculate character offsets in original text
            # We need to find where these tokens start/end in the original
            chars_before = len(encoder.decode(section_tokens[:start]))
            chars_in_chunk = len(chunk_text_content)
            
            chunks.append(Chunk(
                chunk_id=f"{doc_id}::c{chunk_index:04d}",
                doc_id=doc_id,
                doc_path=doc_path,
                text=chunk_text_content,
                token_count=len(chunk_tokens_list),
                start_char=chars_before,
                end_char=chars_before + chars_in_chunk,
                section=section_heading,
                metadata={},
            ))
            chunk_index += 1
            
            if end >= total_tokens:
                break
            start += stride
    
    return chunks


def chunk_file(
    file_path: str | Path,
    chunk_tokens: int = 256,
    chunk_overlap_tokens: int = 32,
    encoding: str = "cl100k_base",
) -> list[Chunk]:
    """Load a markdown file and chunk it."""
    path = Path(file_path)
    doc_id = path.stem  # filename without extension
    
    with open(path) as f:
        text = f.read()
    
    return chunk_text(
        text=text,
        doc_id=doc_id,
        doc_path=str(path.absolute()),
        chunk_tokens=chunk_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        encoding=encoding,
    )


def chunk_corpus(
    corpus_dir: str | Path,
    chunk_tokens: int = 256,
    chunk_overlap_tokens: int = 32,
    encoding: str = "cl100k_base",
) -> list[Chunk]:
    """Chunk all markdown files in a directory."""
    corpus_dir = Path(corpus_dir)
    all_chunks: list[Chunk] = []
    
    for md_file in sorted(corpus_dir.glob("*.md")):
        chunks = chunk_file(
            md_file,
            chunk_tokens=chunk_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            encoding=encoding,
        )
        all_chunks.extend(chunks)
    
    return all_chunks
