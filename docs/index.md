# Ahocorasick NER — Documentation

Fast, dictionary-based Named Entity Recognition using the Aho-Corasick algorithm.

---

## Quick Start

### Installation
```bash
# Core library
uv pip install ahocorasick-ner

# With HuggingFace dataset support
uv pip install ahocorasick-ner[datasets]
```

### Programmatic Usage
```python
from ahocorasick_ner import AhocorasickNER

ner = AhocorasickNER()
ner.add_word("artist", "Metallica")
ner.add_word("artist", "Iron Maiden")
ner.add_word("album", "Master of Puppets")
ner.fit()

for entity in ner.tag("I love Metallica's Master of Puppets"):
    print(entity)
# Output:
# {'start': 7, 'end': 16, 'word': 'Metallica', 'label': 'artist'}
# {'start': 22, 'end': 40, 'word': 'Master of Puppets', 'label': 'album'}
```

---

## Architecture

### Core Components

#### `AhocorasickNER` — `ahocorasick_ner/__init__.py:7`
Main NER engine. Wraps `pyahocorasick.Automaton` with word boundary detection and overlap resolution.

**Key Methods**:
- `add_word(label: str, example: str)` — register entity
- `fit()` — finalize automaton (required before tagging)
- `tag(haystack: str, min_word_len: int = 5)` — find entities
- `save(path)` / `load(path)` — persist automaton via pickle

**Algorithms**:
- **Word Boundary Detection** — regex `\w` before/after match to skip partial words
- **Overlap Resolution** — greedy longest-match-first; ties broken by start position
- **Latency** — O(n) per character after O(m) fit time (m = total entity chars)

#### Dataset Loaders
Pre-built NER systems loaded from HuggingFace:
- **Metal Archives** — `EncyclopediaMetallvmNER` — bands, tracks, albums (≈15K entities)
- **Multi-Genre Music** — `MusicNER` — classical, jazz, prog, trance, metal (≈50K)
- **IMDB** — `ImdbNER` — actors, directors, writers, composers (≈20K)

Each extends `AhocorasickNER` and implements `load_huggingface()`.

---

## Development

### Directory Structure
```
ahocorasick_ner/
  __init__.py           # AhocorasickNER class
  datasets.py           # Preset dataset loaders
  opm.py               # OpenVoiceOS plugin
  version.py           # Version string

docs/
  index.md             # This file

test/
  unittests/
    test_ner.py        # Core NER unit tests
```

### Running Tests
```bash
# Core NER tests
uv run pytest test/unittests/ -v

# With coverage
uv run pytest test/unittests/ --cov=ahocorasick_ner --cov-report=term-missing
```

---

## Performance Notes

### Aho-Corasick Guarantees
- **Fit time**: O(m) where m = sum of entity lengths
- **Match time**: O(n + z) where n = text length, z = number of matches
- **Memory**: ~64 bytes per trie node (heuristic estimate)

### Overlap Handling
Greedy longest-match strategy:
1. Find all overlapping matches
2. Sort by length (descending), then by position
3. Select non-overlapping matches in order

Example: entities ["abc", "bcd", "bcde"] + text "abcde"
- Candidates: (0,2, "abc"), (1,3, "bcd"), (1,4, "bcde")
- Sorted: (1,4, "bcde"), (1,3, "bcd"), (0,2, "abc")
- Selected: (1,4, "bcde") — blocks (1,3) and (0,2) because they overlap

### Estimated RAM
- Base automaton: ~100-500 bytes (overhead)
- Per entity: ~64 bytes per character in trie
- Example: 10K entities × 20 chars avg = ~12.8 MB

Actual memory depends on:
- Automaton compression (shared prefixes)
- Python object overhead
- System allocator fragmentation

---

## Comparison to Alternatives

| Feature | Aho-Corasick | Regex | Fuzzy | Transformer |
|---------|-------------|-------|-------|------------|
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Simplicity | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Setup | Instant | Instant | Requires tuning | Training required |
| Large vocab (10K+) | Best | Slow | Slow | N/A |
| Typo tolerance | None | Possible | Yes | Yes |
| Interpretability | 100% | 100% | ~80% | ~20% |

Use Aho-Corasick when:
- ✅ Large, well-defined vocabularies (1K–1M+ entities)
- ✅ Exact matching with clear labels
- ✅ Low latency required
- ✅ Explainability matters

Use alternatives when:
- ❌ Need fuzzy/typo tolerance
- ❌ Entity boundaries are ambiguous
- ❌ Morphological variation (e.g., plurals)
- ❌ Context-dependent meanings (NER models better)

---

## Troubleshooting

**Q: Tagging is too slow**
- **Cause**: Large vocabulary (>100K entities) or very long text (>10K chars)
- **Fix**: Reduce vocabulary or filter text before tagging

**Q: Word boundaries skip valid entities**
- **Cause**: Entities contain underscores or special characters that `\w` matches
- **Fix**: Adjust `min_word_len` or modify regex in `ahocorasick_ner/__init__.py:91`

**Q: Import jobs never complete**
- **Cause**: HuggingFace dataset not found or network issue
- **Fix**: Check dataset name at huggingface.co/datasets; enable verbose logging

---

## License
Apache 2.0

---

## See Also
- `FAQ.md` — Common questions and gotchas
- `AUDIT.md` — Known issues and design notes
- `SUGGESTIONS.md` — Future enhancements
- `MAINTENANCE_REPORT.md` — Change history
