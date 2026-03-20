# Getting Started

## Installation

### Core Library (pyahocorasick backend)

```bash
uv pip install ahocorasick-ner
```

Or with pip:
```bash
pip install ahocorasick-ner
```

### Optional: NumPy Backend

```bash
uv pip install ahocorasick-ner[numpy]
```

Pure-Python NER without C dependencies. Useful for environments where compiling extensions is difficult.

### Optional: ONNX Backend

```bash
uv pip install ahocorasick-ner[onnx]
```

Export models to ONNX format for edge computing, WASM, and cross-platform deployment.

### Optional: Dataset Support

```bash
uv pip install ahocorasick-ner[datasets]
```

Load pre-built vocabularies from HuggingFace (Metal Archives, Music, IMDB).

### All Features

```bash
uv pip install ahocorasick-ner[numpy,onnx,datasets]
```

---

## Your First NER System — 5 Minutes

### 1. Basic Setup

```python
from ahocorasick_ner import AhocorasickNER

# Create NER system
ner = AhocorasickNER()

# Add entities
ner.add_word("artist", "Metallica")
ner.add_word("artist", "Iron Maiden")
ner.add_word("album", "Master of Puppets")
ner.add_word("album", "The Number of the Beast")

# Finalize automaton
ner.fit()
```

### 2. Extract Entities

```python
text = "I love Metallica's Master of Puppets and Iron Maiden's The Number of the Beast"

for entity in ner.tag(text):
    print(entity)
```

Output:
```python
{'start': 7, 'end': 15, 'word': 'Metallica', 'label': 'artist'}
{'start': 22, 'end': 40, 'word': 'Master of Puppets', 'label': 'album'}
{'start': 46, 'end': 57, 'word': 'Iron Maiden', 'label': 'artist'}
{'start': 62, 'end': 87, 'word': 'The Number of the Beast', 'label': 'album'}
```

### 3. Case Sensitivity

By default, matching is case-insensitive. To enable case sensitivity:

```python
ner = AhocorasickNER(case_sensitive=True)
ner.add_word("name", "Apple")      # Only matches "Apple", not "apple"
ner.add_word("fruit", "apple")      # Different entity, exact case
ner.fit()

ner.tag("I use Apple computers and eat apples")
```

### 4. Save and Load

```python
# After training, save for reuse
ner.save("music_ner.ahocorasick")

# Later, load the trained automaton
ner2 = AhocorasickNER()
ner2.load("music_ner.ahocorasick")

# Ready to use immediately (no re-fitting needed)
list(ner2.tag("Metallica is great"))
```

---

## Typical Workflow

### Pattern 1: One-Shot Tagging

```python
from ahocorasick_ner import AhocorasickNER

ner = AhocorasickNER()
ner.add_word("company", "Microsoft")
ner.add_word("company", "Apple")
ner.fit()

results = list(ner.tag("Microsoft and Apple are tech giants"))
print(results)
```

### Pattern 2: Batch Processing

```python
ner = AhocorasickNER()
# ... add entities and fit ...

documents = [
    "Metallica released Master of Puppets in 1986",
    "Iron Maiden's debut album was revolutionary",
    "Black Sabbath pioneered heavy metal music"
]

for doc in documents:
    for entity in ner.tag(doc):
        print(f"Doc: {doc[:30]}... | Entity: {entity['label']}")
```

### Pattern 3: Entity Management

```python
# Start with an existing automaton
ner = AhocorasickNER()
ner.add_word("artist", "Metallica")
ner.add_word("artist", "Iron Maiden")
ner.fit()

# Use it
entities = list(ner.tag("Metallica and Iron Maiden"))

# Later, add more entities (need to re-fit)
ner.add_word("artist", "Black Sabbath")  # Automatically marks as unfitted
ner.fit()  # Re-finalize

# Continue using
entities = list(ner.tag("Black Sabbath is awesome"))
```

---

## Choosing a Backend

### Default: pyahocorasick (C-based)

**Best for:** Production, performance-critical applications.

```python
from ahocorasick_ner import AhocorasickNER
ner = AhocorasickNER()
```

**Pros:**
- ⭐⭐⭐⭐⭐ Fastest (C implementation)
- Mature, battle-tested algorithm
- Efficient memory usage

**Cons:**
- Requires C compiler to install
- Not available on all platforms (rare)

### NumPy Backend

**Best for:** Pure-Python environments, cross-platform compatibility.

```python
from ahocorasick_ner.numpy_backend import NumpyAhocorasickNER
ner = NumpyAhocorasickNER()
```

**Pros:**
- Pure Python (no C compilation)
- Portable to any platform with NumPy
- Same API as default backend

**Cons:**
- ~10-30% slower than C backend
- Higher memory overhead

### ONNX Backend

**Best for:** Edge computing, WASM, cross-platform ML pipelines.

```python
from ahocorasick_ner.onnx_backend import OnnxAhocorasickNER
ner = OnnxAhocorasickNER()
ner.save("model")  # Creates model.onnx + model.npz
```

**Pros:**
- ONNX standard format (widely supported)
- Deploy to browsers, mobile, Kubernetes
- Works with any ONNX runtime

**Cons:**
- Slightly slower than pyahocorasick
- ONNX file + pickle overhead

See **[Backends](backends.md)** for detailed comparison and benchmarks.

---

## Next Steps

- **[API Reference](api-reference.md)** — All methods and parameters
- **[Examples](examples.md)** — Real-world usage patterns
- **[Datasets](datasets.md)** — Load pre-built vocabularies
- **[Integration](integration.md)** — Use with OpenVoiceOS
- **[Performance](performance.md)** — Optimize for your use case
