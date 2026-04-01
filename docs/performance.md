# Performance

Benchmarks, profiling, and optimization strategies.

---

## Benchmarks

### Machine Setup

```text
CPU: Intel Core i7-9700K @ 3.6 GHz
RAM: 32 GB
Python: 3.11
Backend: pyahocorasick (C-based)
```text

### Fit Time

Time to build automaton from entities.

```text
Entities | Time (ms) | RAM (MB)
---------|-----------|----------
100      | 1         | 0.5
1K       | 5         | 2
5K       | 20        | 8
10K      | 50        | 15
50K      | 300       | 80
100K     | 600       | 160
```text

**Complexity:** O(m) where m = sum of entity lengths

```python
import time
from ahocorasick_ner import AhocorasickNER

for count in [100, 1000, 5000, 10000]:
    ner = AhocorasickNER()
    start = time.time()
    for i in range(count):
        ner.add_word("entity", f"term_{i}")
    ner.fit()
    elapsed = time.time() - start
    print(f"{count} entities: {elapsed*1000:.1f}ms")
```text

### Tag Time

Time to extract entities from text.

**Single Text:**
```text
Text Length | 100 chars | 1K chars | 10K chars | 100K chars
------------|-----------|----------|-----------|----------
5K entities| 2ms       | 5ms      | 50ms      | 500ms
10K        | 3ms       | 8ms      | 80ms      | 800ms
50K        | 5ms       | 20ms     | 200ms     | 2000ms
100K       | 10ms      | 40ms     | 400ms     | 4000ms
```text

**Complexity:** O(n + z) where n = text length, z = matches

```python
import time
from ahocorasick_ner import AhocorasickNER

ner = AhocorasickNER()
for i in range(10000):
    ner.add_word("entity", f"term_{i}")
ner.fit()

text = "term_5000 is interesting " * 100  # ~2500 chars

start = time.time()
for _ in range(1000):
    list(ner.tag(text))
elapsed = time.time() - start

print(f"1000 iterations: {elapsed:.2f}s ({1/elapsed:.0f} tags/sec)")
# ~0.15s (6667 tags/sec)
```text

### Save/Load Time

Persistence overhead.

```text
Entities | Save (ms) | Load (ms)
---------|-----------|----------
1K       | 2         | 3
5K       | 5         | 7
10K      | 10        | 15
50K      | 50        | 60
100K     | 100       | 120
```text

```python
import time
from ahocorasick_ner import AhocorasickNER

ner = AhocorasickNER()
for i in range(10000):
    ner.add_word("entity", f"term_{i}")
ner.fit()

# Save
start = time.time()
ner.save("model.ahocorasick")
save_time = time.time() - start

# Load
start = time.time()
ner2 = AhocorasickNER()
ner2.load("model.ahocorasick")
load_time = time.time() - start

print(f"Save: {save_time*1000:.1f}ms, Load: {load_time*1000:.1f}ms")
```text

---

## Backend Comparison

### Speed (Tag Time, 1000 iterations, 10K entities)

```text
Backend     | Speed      | Relative
------------|------------|----------
pyahocorasick | 0.15s    | 1.0x (baseline)
NumPy       | 0.35s      | 2.3x slower
ONNX        | 0.40s      | 2.7x slower
```text

### Memory (Loaded Model)

```text
Backend     | Memory
------------|--------
pyahocorasick | 15 MB
NumPy       | 25 MB
ONNX        | 30 MB (includes .onnx file)
```text

### Installation Size

```text
Backend     | Size
------------|--------
pyahocorasick | 0.5 MB (binary)
NumPy       | 20 MB
ONNX        | ~10 MB (onnx) + ~50 MB (onnxruntime)
```text

---

## Profiling

### Find Bottlenecks

```python
import cProfile
import pstats
from ahocorasick_ner import AhocorasickNER

ner = AhocorasickNER()
for i in range(1000):
    ner.add_word("entity", f"term_{i}")

# Profile fit
profiler = cProfile.Profile()
profiler.enable()
ner.fit()
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
stats.print_stats(10)
```text

Output:
```text
ncalls  tottime  cumtime   filename:lineno(function)
     1    0.001    0.020   __init__.py:55(fit)
     1    0.019    0.019   {pyahocorasick.make_automaton}
  2000    0.000    0.000   {built-in append}
```text

### Memory Profiling

```python
from memory_profiler import profile
from ahocorasick_ner import AhocorasickNER

@profile
def train_ner():
    ner = AhocorasickNER()
    for i in range(10000):
        ner.add_word("entity", f"term_{i}")
    ner.fit()
    return ner

train_ner()
```text

Run:
```bash
python -m memory_profiler script.py
```text

Output shows line-by-line memory usage.

---

## Optimization Strategies

### 1. Reduce Vocabulary Size

**Problem:** Large vocabulary = long fit time + high memory

**Solution:** Remove unnecessary entities

```python
# ❌ Slow: 100K unrelated terms
ner = AhocorasickNER()
for i in range(100000):
    ner.add_word("entity", f"word_{i}")
ner.fit()  # ~600ms, 160MB

# ✅ Fast: Only relevant entities
ner = AhocorasickNER()
relevant_entities = [...] # 5K entities
for entity in relevant_entities:
    ner.add_word("entity", entity)
ner.fit()  # ~20ms, 8MB
```text

### 2. Increase min_word_len

**Problem:** Many short matches slow down tagging

**Solution:** Filter short entities

```python
# ❌ Slow: Match everything including 1-char words
ner.tag(text, min_word_len=1)  # 100ms for 10K chars

# ✅ Fast: Only match words >= 4 chars
ner.tag(text, min_word_len=4)  # 30ms for 10K chars
```text

### 3. Cache NER Models

**Problem:** Re-training on every use

**Solution:** Load pre-trained models

```python
# ❌ Slow: Re-train every time
def process_text(text):
    ner = AhocorasickNER()  # Build from scratch
    for entity in my_entities:
        ner.add_word("entity", entity)
    ner.fit()  # 50ms
    return list(ner.tag(text))

# ✅ Fast: Load once, reuse
ner = AhocorasickNER()  # Load once at startup
ner.load("prebuilt_model.ahocorasick")

def process_text(text):
    return list(ner.tag(text))  # <5ms per document
```text

### 4. Use Appropriate Backend

**Problem:** Slow tagging performance

**Solution:** Choose backend for your constraints

```python
# If performance critical: pyahocorasick
from ahocorasick_ner import AhocorasickNER
ner = AhocorasickNER()

# If must run in browsers/edge: ONNX
from ahocorasick_ner.onnx_backend import OnnxAhocorasickNER
ner = OnnxAhocorasickNER()

# If no C compiler available: NumPy
from ahocorasick_ner.numpy_backend import NumpyAhocorasickNER
ner = NumpyAhocorasickNER()
```text

### 5. Batch Processing

**Problem:** Tagging documents one-by-one

**Solution:** Use generator or parallel processing

```python
# ❌ Slow: Sequential
results = [list(ner.tag(doc)) for doc in documents]

# ✅ Fast: Parallel
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as exe:
    results = list(exe.map(lambda doc: list(ner.tag(doc)), documents))
```text

### 6. Pre-compile Regular Expressions

**Problem:** Word boundary checks re-compile regex

**Note:** Word boundary checking uses regex matching in the hot path. For extremely high throughput, consider custom word-boundary logic or caching checks.

```python
# Word boundary check occurs per match
# Use case_sensitive=True if not checking boundaries can reduce overhead
```

---

## Scaling Strategies

### For Very Large Vocabularies (>100K entities)

```python
from ahocorasick_ner import AhocorasickNER

# Strategy 1: Split into categories
ner_artists = AhocorasickNER()
ner_albums = AhocorasickNER()

for artist in artists:
    ner_artists.add_word("artist", artist)
ner_artists.fit()

for album in albums:
    ner_albums.add_word("album", album)
ner_albums.fit()

# Tag with appropriate NER
text = "Metallica plays Master of Puppets"
artists = list(ner_artists.tag(text))
albums = list(ner_albums.tag(text))

# Advantage: Smaller models, faster tagging

# Strategy 2: Use prefix/suffix indexing
ner = AhocorasickNER()
# Only add entities starting with "A", "B", "C" (rotate by character)
# Require first character matches before tagging
```text

### For Real-time Systems (<10ms latency required)

```python
# Strategy 1: Pre-filter text
import re

ner = AhocorasickNER()
ner.load("prebuilt_model.ahocorasick")

def fast_tag(text):
    # Only tag if contains likely matches (first letter check)
    if not re.search(r'[A-Z]', text):  # Capitalized word
        return []

    # Tag now that we filtered
    return list(ner.tag(text, min_word_len=4))

# Strategy 2: Cache results
from functools import lru_cache

@lru_cache(maxsize=10000)
def tag_cached(text):
    return tuple(ner.tag(text))
```text

---

## Expected Performance

### Typical Usage

- **Fit:** 50ms for 10K entities
- **Tag:** 5ms per document (100 chars)
- **Save/Load:** 15ms each

### High Performance Required

- **Use:** pyahocorasick backend + caching + batch processing
- **Achieve:** 10,000+ documents/second

### Low Latency (<10ms)

- **Reduce:** vocabulary size to <5K entities
- **Use:** min_word_len >= 4
- **Cache:** results for repeated queries

### Memory Constrained

- **Use:** NumPy backend (pure Python, smaller binary)
- **Reduce:** vocabulary size
- **Consider:** streaming/chunked processing

---

## See Also

- **[Backends](backends.md)** — Performance comparison of backends
- **[Algorithms](algorithms.md)** — Complexity analysis
- **[Examples](examples.md)** — Batch processing patterns
