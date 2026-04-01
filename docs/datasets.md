# Pre-built Datasets

Load curated entity vocabularies from HuggingFace without manual training.

---

## Overview

Three pre-built datasets are available, each with ~15K–50K entities:

| Dataset | Entities | Domain | API |
|---------|----------|--------|-----|
| **Encyclopedia Metallum** | ~15K | Metal music | `EncyclopediaMetallvmNER` |
| **Multi-Genre Music** | ~50K | All music genres | `MusicNER` |
| **IMDB** | ~20K | Entertainment | `ImdbNER` |

All three extend `AhocorasickNER` with identical API — just different pre-loaded vocabularies.

---

## Installation

```bash
uv pip install ahocorasick-ner[datasets]
```

Requires: `datasets` library from HuggingFace

---

## Encyclopedia Metallum NER

Metal music entities — bands, tracks, albums, genres.

**Class:** `EncyclopediaMetallvmNER` — `ahocorasick_ner/datasets.py:14`

**Entities (~15K):**
- `artist_name` — Band names (e.g., "Metallica", "Iron Maiden")
- `track_name` — Song titles (e.g., "Master of Puppets")
- `album_name` — Album names (e.g., "Paranoid")
- `album_type` — Album type (e.g., "Full-length", "EP", "Live")
- `music_genre` — Genre tags (e.g., "Thrash Metal", "Heavy Metal")
- `record_label` — Record label names

**Data Source:**
- Dataset: `Jarbas/metal-archives-tracks` (HuggingFace)
- Dataset: `Jarbas/metal-archives-bands` (HuggingFace)

### Basic Usage

```python
from ahocorasick_ner.datasets import EncyclopediaMetallvmNER

# First run: downloads data, trains, saves (slow)
ner = EncyclopediaMetallvmNER(path="metal_ner.ahocorasick")

# Later runs: loads from disk (fast)
ner = EncyclopediaMetallvmNER(path="metal_ner.ahocorasick")

# Use
text = "Metallica's Master of Puppets is a thrash metal masterpiece"
for entity in ner.tag(text):
    print(f"{entity['word']} ({entity['label']})")

# Output:
# Metallica (artist_name)
# Master of Puppets (album_name)
# thrash metal (music_genre)
```

### Constructor

```python
EncyclopediaMetallvmNER(
    path: Optional[str] = None,
    case_sensitive: bool = False
)
```

**Parameters:**
- `path` (str, optional) — Path to saved automaton file
  - If provided AND file exists: loads from disk (instant)
  - If provided AND file doesn't exist: trains and saves (then loads)
  - If not provided: trains in memory only
- `case_sensitive` (bool) — Case-sensitive matching (default: False)

**Example:**
```python
# Load from disk or create
ner = EncyclopediaMetallvmNER(path="my_model.ahocorasick")

# In-memory only (no save)
ner = EncyclopediaMetallvmNER()

# Case-sensitive
ner = EncyclopediaMetallvmNER(case_sensitive=True)
```

---

## Music NER

Comprehensive multi-genre music vocabulary.

**Class:** `MusicNER` — `ahocorasick_ner/datasets.py:69`

**Entities (~50K):**
- All entities from Encyclopedia Metallum
- Plus:
  - Jazz artists, standards, composers
  - Progressive rock bands and albums
  - Classical composers and works
  - Trance producers and tracks

**Data Sources:**
- Metal Archives (same as above)
- Jazz and classical datasets
- Electronic music datasets

### Basic Usage

```python
from ahocorasick_ner.datasets import MusicNER

ner = MusicNER(path="music_ner.ahocorasick")

text = "John Coltrane played A Love Supreme, while Metallica recorded Master of Puppets"
for entity in ner.tag(text):
    print(f"{entity['word']} ({entity['label']})")

# Output:
# John Coltrane (artist_name)
# A Love Supreme (album_name)
# Metallica (artist_name)
# Master of Puppets (album_name)
```

### Constructor

```python
MusicNER(
    path: Optional[str] = None,
    case_sensitive: bool = False
)
```

Identical to `EncyclopediaMetallvmNER`.

---

## IMDB NER

Entertainment industry entities.

**Class:** `ImdbNER` — `ahocorasick_ner/datasets.py:167`

**Entities (~20K):**
- `actor_name` — Actor names
- `director_name` — Director names
- `writer_name` — Screenwriter names
- `composer_name` — Film composer names
- `movie_title` — Movie titles
- `studio_name` — Film studio names

**Data Source:**
- Dataset: `Jarbas/imdb-entities` (HuggingFace)

### Basic Usage

```python
from ahocorasick_ner.datasets import ImdbNER

ner = ImdbNER(path="imdb_ner.ahocorasick")

text = "Directed by Steven Spielberg, featuring Tom Cruise in a thriller by Hans Zimmer"
for entity in ner.tag(text):
    print(f"{entity['word']} ({entity['label']})")

# Output:
# Steven Spielberg (director_name)
# Tom Cruise (actor_name)
# Hans Zimmer (composer_name)
```

### Constructor

```python
ImdbNER(
    path: Optional[str] = None,
    case_sensitive: bool = False
)
```

Identical to `EncyclopediaMetallvmNER`.

---

## Working with Datasets

### Combining Multiple Datasets

```python
from ahocorasick_ner import AhocorasickNER
from ahocorasick_ner.datasets import EncyclopediaMetallvmNER, ImdbNER

# Create combined NER system
ner = AhocorasickNER()

# Load first dataset
metal_ner = EncyclopediaMetallvmNER()
# (Can't directly merge, so instead create fresh NER)

# Actually: load datasets sequentially
ner = AhocorasickNER()
# Manually add entities from both sources
# Or: use one pre-built dataset that's comprehensive enough

# Better: use MusicNER (combines multiple genres)
ner = MusicNER()  # 50K entities across all music genres
```

### Saving Loaded Datasets

```python
from ahocorasick_ner.datasets import MusicNER

# Load (might be first time, downloads data)
ner = MusicNER()

# Save to disk for later
ner.save("music_ner.ahocorasick")

# Next time: instant load
ner2 = MusicNER(path="music_ner.ahocorasick")
```

### Switching Between Datasets

```python
from ahocorasick_ner.datasets import EncyclopediaMetallvmNER, ImdbNER

# Metal only
metal_ner = EncyclopediaMetallvmNER()
print(metal_ner.tag("Metallica rocks"))

# Entertainment only
imdb_ner = ImdbNER()
print(imdb_ner.tag("Tom Cruise starred"))

# Choose based on domain:
def get_ner(domain):
    if domain == "music":
        return EncyclopediaMetallvmNER()
    elif domain == "entertainment":
        return ImdbNER()
    else:
        raise ValueError(f"Unknown domain: {domain}")
```

---

## Performance with Large Datasets

### Load Time

```python
import time
from ahocorasick_ner.datasets import MusicNER

# First load: trains from HuggingFace (~30-60s)
start = time.time()
ner = MusicNER()
print(f"First load: {time.time() - start:.1f}s")
# First load: 45.3s

# Save for reuse
ner.save("music_ner.ahocorasick")

# Second load: instant (from disk)
start = time.time()
ner = MusicNER(path="music_ner.ahocorasick")
print(f"Reload: {time.time() - start:.1f}s")
# Reload: 0.3s
```

### Tagging Speed

```python
from ahocorasick_ner.datasets import MusicNER
import time

ner = MusicNER(path="music_ner.ahocorasick")

# Tag single document
text = "Metallica performed Master of Puppets"
start = time.time()
entities = list(ner.tag(text))
elapsed = time.time() - start
print(f"Tag 1 doc: {elapsed*1000:.1f}ms")
# Tag 1 doc: 2.3ms

# Tag 1000 documents
texts = [text] * 1000
start = time.time()
for t in texts:
    list(ner.tag(t))
elapsed = time.time() - start
print(f"Tag 1000 docs: {elapsed:.2f}s ({1/elapsed*1000:.0f} docs/sec)")
# Tag 1000 docs: 2.30s (435 docs/sec)
```

### Memory Usage

```python
import psutil
import os

from ahocorasick_ner.datasets import MusicNER

ner = MusicNER()

# Check memory
process = psutil.Process(os.getpid())
mem = process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory: {mem:.1f} MB")
# Memory: 120.5 MB
```

---

## Customization

### Extend a Dataset

```python
from ahocorasick_ner.datasets import MusicNER

class CustomMusicNER(MusicNER):
    def __init__(self, path=None, case_sensitive=False):
        super().__init__(path, case_sensitive)

        # Add custom entities after loading
        self.add_word("artist", "Custom Band")
        self.add_word("album", "Custom Album")
        self.fit()

# Use
ner = CustomMusicNER()
print(ner.tag("Listen to Custom Band's Custom Album"))
```

### Mix Dataset with Custom Entities

```python
from ahocorasick_ner import AhocorasickNER
from ahocorasick_ner.datasets import EncyclopediaMetallvmNER

# Start with pre-built
ner = EncyclopediaMetallvmNER()

# Add custom entities
ner.add_word("city", "New York")
ner.add_word("city", "London")
ner.fit()

# Use combined
text = "Metallica played Master of Puppets in New York"
print(list(ner.tag(text)))
```

---

## Troubleshooting

### Dataset Download Fails

```
HuggingFaceHub API: Expected response [200], got [403]
```

**Solution:** Set HuggingFace token:
```bash
huggingface-cli login
# Enter your token: hf_xxxxx...
```

Or set environment variable:
```bash
export HF_TOKEN="hf_xxxxx..."
```

### Out of Memory

If loading datasets fails due to memory:

```python
# Use smaller dataset
from ahocorasick_ner.datasets import EncyclopediaMetallvmNER
ner = EncyclopediaMetallvmNER()  # ~15K entities, ~50MB

# Not:
from ahocorasick_ner.datasets import MusicNER
ner = MusicNER()  # ~50K entities, ~150MB
```

### Slow First Load

Datasets are downloaded and trained on first use (slow). Save to disk to speed up reuse:

```python
from ahocorasick_ner.datasets import MusicNER

# First time: slow
ner = MusicNER(path="music_ner.ahocorasick")

# Now fast for future runs
ner = MusicNER(path="music_ner.ahocorasick")
```

---

## See Also

- **[API Reference](api-reference.md)** — Full method documentation
- **[Backends](backends.md)** — Choosing backend for large datasets
- **[Performance](performance.md)** — Benchmarks with large vocabularies
- **[Examples](examples.md)** — Using datasets in real applications
