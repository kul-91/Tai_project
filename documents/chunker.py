import re
from typing import Any, Optional, Union

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100

# Resume-like section headings to detect for structure-aware chunking
_SECTION_HEADINGS = re.compile(
    r'^(education|skills|technical\s+skills|projects?|experience|work\s+experience|'
    r'internship|internships|certifications?|positions?\s+of\s+responsibility|'
    r'achievements?|awards?|publications?|languages?|references?|summary|objective|profile)',
    re.IGNORECASE
)

# Detect separator patterns used in resumes: bullets, em-dash, en-dash, section dividers
_BULLET_PATTERN = re.compile(r'^[\u2022\u2023\u25e6\u2043\u2219\-\*]\s+')
_RESUME_SEPARATOR = re.compile(r'[\u2014\u2013]')  # em-dash, en-dash


def _is_section_heading(line: str) -> bool:
    """Returns True if line looks like a resume section heading."""
    stripped = line.strip()
    return bool(_SECTION_HEADINGS.match(stripped))


def _is_markdown_heading(line: str) -> bool:
    return line.strip().startswith('#')


def _split_text_into_units(text: str, max_unit_size: int) -> list[str]:
    """Splits text into structure-aware semantic units.

    Preserves section headings, project titles, bullet points, and paragraph
    boundaries. For resume-like content, detects common section headers and
    keeps project title lines associated with their descriptions.
    Handles PDF em-dash artifacts from text extraction.
    """
    if not text or not text.strip():
        return []

    # Split into lines preserving structure
    lines = text.split('\n')
    raw_units: list[str] = []
    current_block: list[str] = []

    def flush_block():
        """Finalize accumulated block into raw units."""
        if not current_block:
            return
        block_text = ' '.join(l.strip() for l in current_block if l.strip())
        if block_text.strip():
            raw_units.append(block_text)
        current_block.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            # Blank line = paragraph boundary: flush current block
            flush_block()
            i += 1
            continue

        if _is_markdown_heading(stripped):
            flush_block()
            # Extract heading and look ahead for content on same level
            heading_text = stripped
            raw_units.append(heading_text)
            i += 1
            # Collect content lines after heading until next blank / heading
            content_lines = []
            while i < len(lines):
                next_stripped = lines[i].strip()
                if not next_stripped:
                    break
                if _is_markdown_heading(next_stripped) or _is_section_heading(next_stripped):
                    break
                content_lines.append(next_stripped)
                i += 1
            if content_lines:
                current_block.extend(content_lines)
            continue

        if _is_section_heading(stripped):
            flush_block()
            # Emit the section heading as its own unit
            raw_units.append(stripped)
            i += 1
            continue

        # Check for resume project title pattern (line that contains em-dash or en-dash
        # and looks like a heading with technologies)
        if _RESUME_SEPARATOR.search(stripped) and len(stripped) < 200:
            # This is likely a project/experience title line with em-dashes
            flush_block()
            raw_units.append(stripped)
            i += 1
            # Collect following bullet/description lines as a block  
            while i < len(lines):
                next_stripped = lines[i].strip()
                if not next_stripped:
                    break
                if _is_section_heading(next_stripped) or _is_markdown_heading(next_stripped):
                    break
                # If it's another em-dash line (new project), also break
                if _RESUME_SEPARATOR.search(next_stripped) and len(next_stripped) < 200:
                    break
                current_block.append(next_stripped)
                i += 1
            flush_block()
            continue

        # Regular content line: accumulate into block
        current_block.append(stripped)
        i += 1

    flush_block()

    # Now split each raw unit into sentences where necessary
    sentence_units: list[str] = []
    for unit in raw_units:
        sentences = re.split(r'(?<=[.!?])\s+', unit)
        for s in sentences:
            s_clean = s.strip()
            if s_clean:
                sentence_units.append(s_clean)

    # Handle oversized units: split at word boundaries
    final_units: list[str] = []
    for unit in sentence_units:
        if len(unit) <= max_unit_size:
            final_units.append(unit)
        else:
            words = unit.split()
            current_group: list[str] = []
            current_len = 0

            for word in words:
                word_len = len(word)
                # Fallback to character slicing only if a single word exceeds max_unit_size
                if word_len > max_unit_size:
                    if current_group:
                        final_units.append(' '.join(current_group))
                        current_group = []
                        current_len = 0
                    for idx in range(0, word_len, max_unit_size):
                        final_units.append(word[idx: idx + max_unit_size])
                    continue

                needed = word_len + (1 if current_group else 0)
                if current_len + needed <= max_unit_size:
                    current_group.append(word)
                    current_len += needed
                else:
                    final_units.append(' '.join(current_group))
                    current_group = [word]
                    current_len = word_len

            if current_group:
                final_units.append(' '.join(current_group))

    return final_units


class TextChunker:
    """Splits document text records into structure-aware, sentence-level overlapping chunks.

    Improvements over naive character-slicing:
    - Preserves section headings, project titles, and bullet points together.
    - Detects resume sections (Education, Projects, Experience, Skills, etc.).
    - Handles em-dash separator patterns from PDF text extraction.
    - Splits between projects/sections rather than mid-description.
    - Complete word and sentence preservation with configurable overlap.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP
    ) -> None:
        """Initialize TextChunker with configurable chunk_size and overlap.

        Args:
            chunk_size: Target character limit per chunk. Must be > 0.
            overlap: Target character overlap between consecutive chunks. Must be >= 0 and < chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError('chunk_size must be a positive integer.')
        if overlap < 0:
            raise ValueError('overlap must be a non-negative integer.')
        if overlap >= chunk_size:
            raise ValueError('overlap must be strictly less than chunk_size.')

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        """Chunk a single document record into structure-aware segments.

        Args:
            doc: Document record containing 'text', 'source', and optional 'page'.

        Returns:
            List of chunk records with preserved metadata.
        """
        text = doc.get('text', '')
        if not text or not text.strip():
            return []

        source = doc.get('source', '')
        page = doc.get('page', None)

        units = _split_text_into_units(text, self.chunk_size)
        if not units:
            return []

        chunks: list[dict[str, Any]] = []
        current_units: list[str] = []
        current_len = 0

        for unit in units:
            unit_len = len(unit)

            if not current_units:
                current_units.append(unit)
                current_len = unit_len
            else:
                is_heading = _is_markdown_heading(unit) or _is_section_heading(unit)
                sep = '\n' if is_heading else ' '
                needed = len(sep) + unit_len

                if current_len + needed <= self.chunk_size:
                    current_units.append(unit)
                    current_len += needed
                else:
                    # Finalize current chunk
                    chunk_str = self._format_chunk_units(current_units)
                    if chunk_str.strip():
                        chunks.append({
                            'text': chunk_str,
                            'source': source,
                            'page': page
                        })

                    # Calculate semantic overlap (complete previous sentences/units)
                    overlap_units: list[str] = []
                    acc_overlap = 0
                    for u in reversed(current_units):
                        needed_ov = len(u) + (1 if overlap_units else 0)
                        if acc_overlap + needed_ov <= self.overlap:
                            overlap_units.insert(0, u)
                            acc_overlap += needed_ov
                        else:
                            break

                    # Ensure overlap_units + unit fits within chunk_size
                    while overlap_units:
                        cand_len = (
                            sum(len(u) for u in overlap_units)
                            + (len(overlap_units) - 1)
                            + 1 + unit_len
                        )
                        if cand_len <= self.chunk_size:
                            break
                        overlap_units.pop(0)

                    current_units = overlap_units + [unit]
                    current_len = (
                        sum(len(u) for u in current_units)
                        + (len(current_units) - 1 if current_units else 0)
                    )

        if current_units:
            chunk_str = self._format_chunk_units(current_units)
            if chunk_str.strip():
                chunks.append({
                    'text': chunk_str,
                    'source': source,
                    'page': page
                })

        return chunks

    def _format_chunk_units(self, units: list[str]) -> str:
        if not units:
            return ''
        result = []
        for i, u in enumerate(units):
            if i > 0:
                if _is_markdown_heading(u) or _is_section_heading(u):
                    result.append('\n' + u)
                else:
                    result.append(' ' + u)
            else:
                result.append(u)
        return ''.join(result)

    def chunk_documents(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Chunk a list of document records while keeping page boundaries intact."""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


def chunk_document(
    doc: dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> list[dict[str, Any]]:
    """Helper to chunk a single document record with structure-aware semantic chunking."""
    return TextChunker(chunk_size=chunk_size, overlap=overlap).chunk_document(doc)


def chunk_documents(
    docs: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> list[dict[str, Any]]:
    """Helper to chunk a list of document records."""
    return TextChunker(chunk_size=chunk_size, overlap=overlap).chunk_documents(docs)
