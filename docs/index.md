# ahocorasick-ner Documentation

**Fast, dictionary-based Named Entity Recognition using the Aho-Corasick algorithm.**

This library excels at extracting known entities from text with zero machine learning overhead. Perfect for rule-based systems, knowledge graphs, and high-performance pipelines.

---

## Quick Links

- **[Getting Started](getting-started.md)** — Installation and 5-minute quickstart
- **[API Reference](api-reference.md)** — Complete method signatures and usage
- **[Backends](backends.md)** — Pyahocorasick, NumPy, ONNX comparison
- **[Algorithms](algorithms.md)** — How Aho-Corasick works, word boundaries, overlap resolution
- **[Examples](examples.md)** — Real-world usage patterns
- **[Integration](integration.md)** — OpenVoiceOS plugin setup
- **[Datasets](datasets.md)** — Using pre-built entity loaders (Metal, Music, IMDB)
- **[Performance](performance.md)** — Benchmarks, profiling, optimization
- **[Troubleshooting](troubleshooting.md)** — Common issues and solutions

---

## What is Aho-Corasick NER?

The Aho-Corasick algorithm is a finite state machine for multi-pattern string matching:

| Property | Value |
|----------|-------|
| **Fit time** | O(m) — m = sum of entity lengths |
| **Match time** | O(n + z) — n = text length, z = matches |
| **Setup** | Instant (no training, no ML) |
| **Vocab size** | 1K–1M+ entities |
| **Accuracy** | 100% for exact matches |

**When to use:**
- ✅ Large, well-defined vocabularies
- ✅ Exact matching with clear labels
- ✅ Low latency required
- ✅ Explainability matters (no black box)

**When NOT to use:**
- ❌ Need fuzzy/typo tolerance → use `rapidfuzz`
- ❌ Morphological variation (plurals, tenses) → use NLP models
- ❌ Context-dependent meanings → use transformers

---

## Core Concepts

### Automaton
A finite state machine (FSM) built from entities. Once finalized via `fit()`, searches are O(n) regardless of vocabulary size.

```python
ner = AhocorasickNER()
ner.add_word("city", "New York")
ner.add_word("city", "London")
ner.fit()  # Build FSM
```

### Word Boundaries
By default, matches must respect word boundaries (alphanumeric before/after). This prevents "New York" from matching inside "NewYork_Inc".

```python
ner.tag("I visited NewYork_Inc in New York")
# Only matches "New York", not inside "NewYork_Inc"
```

### Overlap Resolution
When entities overlap, the library uses greedy longest-match-first strategy:

```python
ner.add_word("entity", "abc")
ner.add_word("entity", "bcd")
ner.add_word("entity", "bcde")
ner.fit()

# In text "abcde":
# Candidates: "abc", "bcd", "bcde"
# Selected: "bcde" (longest; blocks overlapping matches)
```

---

## Three Backends

| Backend | Use Case | Speed | Dependencies |
|---------|----------|-------|--------------|
| **pyahocorasick** | Production (default) | ⭐⭐⭐⭐⭐ Fastest | C extension (compiles native) |
| **NumPy** | Portable | ⭐⭐⭐⭐ | Pure Python, requires numpy |
| **ONNX** | Edge/WASM | ⭐⭐⭐⭐ | onnxruntime (cross-platform) |

All three share the same API — switch backends without code changes:

```python
from ahocorasick_ner import AhocorasickNER          # pyahocorasick
from ahocorasick_ner.numpy_backend import NumpyAhocorasickNER
from ahocorasick_ner.onnx_backend import OnnxAhocorasickNER
```

See **[Backends](backends.md)** for detailed comparison.

---

## Pre-built Datasets

Load curated entity vocabularies from HuggingFace:

```python
from ahocorasick_ner.datasets import EncyclopediaMetallvmNER, MusicNER, ImdbNER

# Metal Archives (bands, tracks, albums — ~15K entities)
metal_ner = EncyclopediaMetallvmNER()
metal_ner.tag("Metallica played Master of Puppets")

# Multi-genre music (classical, jazz, prog, trance, metal — ~50K)
music_ner = MusicNER()

# IMDB (actors, directors, writers, composers — ~20K)
imdb_ner = ImdbNER()
```

See **[Datasets](datasets.md)** for full documentation.

---

## OpenVoiceOS Integration

Automatically register entities and perform NER on OVOS utterances:

```python
# In your skill:
self.register_entity("artist_name", ["Metallica", "Iron Maiden"])

# OVOS Transformer plugin listens for registration and injects matches
# into the intent match context — available in your intent handler
```

See **[Integration](integration.md)** for setup and examples.

---

## Performance Expectations

### Fit Time
- 100 entities: ~1 ms
- 10K entities: ~50 ms
- 100K entities: ~500 ms

### Match Time
- Text length 100 chars: <1 ms
- Text length 10K chars: 5–10 ms
- O(n) complexity regardless of vocabulary size

### RAM Estimate
- Heuristic: ~64 bytes per trie node
- 10K entities × 20 chars average: ~12.8 MB
- Actual: system-dependent (Python overhead, compression, allocator)

See **[Performance](performance.md)** for benchmarks and profiling.

---

## License

Apache 2.0 — free for commercial and non-commercial use.

---

## Getting Help

- **Quick answers**: See [Troubleshooting](troubleshooting.md)
- **API details**: See [API Reference](api-reference.md)
- **Usage patterns**: See [Examples](examples.md)
- **Internals**: See [Algorithms](algorithms.md)
