# API Reference

Complete documentation of all classes and methods.

---

## AhocorasickNER

Main class for dictionary-based Named Entity Recognition — `ahocorasick_ner/__init__.py:7`.

### Constructor

```python
AhocorasickNER(case_sensitive: bool = False)
```text

**Parameters:**
- `case_sensitive` (bool) — If `False` (default), all matching is case-insensitive. If `True`, matching respects case.

**Example:**
```python
# Case-insensitive (default)
ner = AhocorasickNER()
ner.add_word("city", "New York")
ner.fit()
list(ner.tag("I visited NEW YORK"))  # Matches

# Case-sensitive
ner = AhocorasickNER(case_sensitive=True)
ner.add_word("city", "New York")
ner.fit()
list(ner.tag("I visited NEW YORK"))  # Does NOT match (wrong case)
```text

---

## Methods

### add_word

```python
add_word(label: str, example: str) -> None
```text

Register an entity for recognition.

**Parameters:**
- `label` (str) — Entity type/category (e.g., "artist", "city", "company")
- `example` (str) — Text to recognize as this entity (e.g., "Metallica")

**Behavior:**
- Adds the entity to the automaton
- Marks automaton as unfitted (must call `fit()` again)
- Case handling determined by `case_sensitive` flag

**Example:**
```python
ner = AhocorasickNER()
ner.add_word("artist", "Metallica")
ner.add_word("artist", "Iron Maiden")
ner.add_word("album", "Master of Puppets")
ner.fit()  # Required before tagging
```text

---

### fit

```python
fit() -> None
```text

Finalize the Aho-Corasick automaton. Must be called after adding words before tagging.

**Behavior:**
- Builds the finite state machine (FSM)
- O(m) time complexity where m = sum of entity lengths
- Can be called multiple times (idempotent if no new words added)
- Automatically called by `tag()` if not already fitted

**Example:**
```python
ner = AhocorasickNER()
ner.add_word("city", "New York")
ner.add_word("city", "London")
ner.fit()  # Build FSM

# Can call tag now
entities = list(ner.tag("I visited New York and London"))

# Add more entities
ner.add_word("city", "Paris")  # Marks as unfitted
ner.fit()  # Must re-fit

entities = list(ner.tag("I visited Paris"))
```text

---

### tag

```python
tag(haystack: str, min_word_len: int = 5) -> Iterable[Dict[str, Union[int, str]]]
```text

Extract entities from text.

**Parameters:**
- `haystack` (str) — Text to search for entities
- `min_word_len` (int, default 5) — Minimum match length in characters

**Returns:**
Generator yielding dictionaries with keys:
- `"start"` (int) — Start position in original text
- `"end"` (int) — End position in original text (inclusive)
- `"word"` (str) — Matched text (preserves original case)
- `"label"` (str) — Entity type

**Behavior:**
- Automatically calls `fit()` if needed
- Respects word boundaries (see [Algorithms](algorithms.md))
- Uses greedy longest-match-first for overlapping entities
- Returns matches in text order (by start position)

**Example:**
```python
ner = AhocorasickNER()
ner.add_word("artist", "Metallica")
ner.add_word("album", "Master of Puppets")
ner.fit()

text = "I love Metallica's Master of Puppets"
for entity in ner.tag(text):
    print(f"Found: {entity['word']} ({entity['label']})")

# Output:
# Found: Metallica (artist)
# Found: Master of Puppets (album)
```text

**Word Length Filter:**
```python
ner.add_word("name", "Jo")   # 2 characters
ner.add_word("name", "John") # 4 characters
ner.fit()

# Default min_word_len=5, so no matches
list(ner.tag("Jo and John arrived"))  # []

# With min_word_len=2
list(ner.tag("Jo and John arrived", min_word_len=2))
# [{'start': 0, 'end': 1, 'word': 'Jo', 'label': 'name'},
#  {'start': 8, 'end': 11, 'word': 'John', 'label': 'name'}]
```text

---

### save

```python
save(path: str) -> None
```text

Save the trained automaton to disk.

**Parameters:**
- `path` (str) — File path where automaton will be saved

**Behavior:**
- Uses pickle to serialize the `pyahocorasick.Automaton`
- Can be called only after `fit()`
- File format is binary (not human-readable)

**Example:**
```python
ner = AhocorasickNER()
ner.add_word("artist", "Metallica")
ner.fit()
ner.save("metal_ner.ahocorasick")

# Later, in another process:
ner2 = AhocorasickNER()
ner2.load("metal_ner.ahocorasick")
list(ner2.tag("Metallica rocks"))
```text

---

### load

```python
load(path: str) -> None
```text

Load a previously saved automaton from disk.

**Parameters:**
- `path` (str) — File path to load from

**Behavior:**
- Reads pickle file and reconstructs automaton
- Overwrites any existing automaton in this instance
- Ready to use immediately (no re-fitting needed)
- Preserves original `case_sensitive` setting from when saved

**Example:**
```python
ner = AhocorasickNER()
ner.load("metal_ner.ahocorasick")  # Load pre-trained

# Use immediately
entities = list(ner.tag("Metallica is awesome"))
print(entities)
```text

---

## NumpyAhocorasickNER

Pure-Python backend using NumPy arrays — `ahocorasick_ner/numpy_backend.py:1`.

**API identical to `AhocorasickNER`:**

```python
from ahocorasick_ner.numpy_backend import NumpyAhocorasickNER

ner = NumpyAhocorasickNER(case_sensitive=False)
ner.add_word("artist", "Metallica")
ner.fit()
list(ner.tag("I like Metallica"))
```text

**Key Differences:**
- Save format: `.npz` (NumPy compressed array) instead of pickle
- No C compiler required
- ~10-30% slower than pyahocorasick
- Cross-platform compatible

---

## OnnxAhocorasickNER

ONNX-compatible backend for edge deployment — `ahocorasick_ner/onnx_backend.py:1`.

**API identical to `AhocorasickNER`:**

```python
from ahocorasick_ner.onnx_backend import OnnxAhocorasickNER

ner = OnnxAhocorasickNER(case_sensitive=False)
ner.add_word("artist", "Metallica")
ner.fit()
list(ner.tag("I like Metallica"))
```text

**Key Differences:**
- Save format: Two files — `model.onnx` + `model.npz`
- Deploy to browsers, mobile, edge devices
- Works with any ONNX runtime (TensorFlow, PyTorch, etc.)
- Slightly slower than pyahocorasick but highly portable

---

## Dataset Classes

Pre-built entity vocabularies from HuggingFace.

### EncyclopediaMetallvmNER

```python
from ahocorasick_ner.datasets import EncyclopediaMetallvmNER

ner = EncyclopediaMetallvmNER(path=None, case_sensitive=False)
```text

Pre-loaded with Metal Archives data (~15K entities):
- `artist_name` — Band names
- `track_name` — Song titles
- `album_name` — Album titles
- `album_type` — Album type (e.g., "Full-length", "EP")
- `music_genre` — Genre tags
- `record_label` — Record labels

**Parameters:**
- `path` (str, optional) — Path to saved automaton. If provided and exists, loads it. Otherwise trains from HuggingFace.
- `case_sensitive` (bool) — Case sensitivity flag

**Example:**
```python
# First run: downloads data, trains, saves
ner = EncyclopediaMetallvmNER(path="metal_ner.ahocorasick")

# Later runs: loads from disk (instant)
ner = EncyclopediaMetallvmNER(path="metal_ner.ahocorasick")

# Use immediately
entities = list(ner.tag("Metallica and Black Sabbath defined metal"))
```text

---

### MusicNER

```python
from ahocorasick_ner.datasets import MusicNER

ner = MusicNER(path=None, case_sensitive=False)
```text

Multi-genre music NER (~50K entities):
- Metal Archives (bands, tracks, albums)
- Jazz artists and standards
- Progressive rock
- Classical composers and works
- Trance producers and tracks

---

### ImdbNER

```python
from ahocorasick_ner.datasets import ImdbNER

ner = ImdbNER(path=None, case_sensitive=False)
```text

IMDB entertainment data (~20K entities):
- Actors
- Directors
- Writers
- Composers

---

## OpenVoiceOS Integration

### AhocorasickNERTransformer

IntentTransformer plugin for OVOS — `ahocorasick_ner/opm.py:16`.

**Event Listeners:**

Automatically bound to:
- `padatious:register_entity` — Listen for entity registrations from skills

**Usage in OVOS Skill:**

```python
class MySkill(OVOSSkill):
    def initialize(self):
        # Register entities — plugin listens automatically
        self.register_entity("artist_name", ["Metallica", "Iron Maiden"])
        self.register_entity("album", ["Master of Puppets", "The Number of the Beast"])

    def handle_music_intent(self, message):
        # Matched entities available in message context
        entities = message.data.get("entities", [])
        for entity in entities:
            print(f"Recognized: {entity['word']} ({entity['label']})")
```text

See **[Integration](integration.md)** for full setup guide.

---

## Return Value Format

All tagging methods return entities as dictionaries:

```python
{
    "start": 7,                  # Start index in original text
    "end": 15,                   # End index (inclusive)
    "word": "Metallica",         # Matched text (preserves original case)
    "label": "artist"            # Entity label from add_word()
}
```text

**Indices are 0-based:**
```python
text = "I love Metallica"
#       0123456789...
entity["start"] = 7
entity["end"] = 15
text[entity["start"]:entity["end"]+1]  # "Metallica"
```text

---

## Error Handling

### FileNotFoundError

Raised by `load()` if path doesn't exist:

```python
try:
    ner.load("nonexistent.ahocorasick")
except FileNotFoundError:
    print("Model file not found")
```text

### ImportError

Raised by dataset classes if HuggingFace `datasets` library not installed:

```python
# Requires: uv pip install ahocorasick-ner[datasets]
from ahocorasick_ner.datasets import MusicNER  # ImportError if datasets not installed
```text

---

## Performance Characteristics

| Operation | Complexity | Time |
|-----------|-----------|------|
| `add_word()` | O(1) amortized | <1 μs |
| `fit()` | O(m) | ~50 ms for 10K entities |
| `tag()` | O(n + z) | ~5 ms for 10K chars |
| `save()` | O(m) | ~10 ms for 10K entities |
| `load()` | O(m) | ~10 ms for 10K entities |

See **[Performance](performance.md)** for detailed benchmarks.

---

## See Also

- **[Algorithms](algorithms.md)** — How Aho-Corasick works internally
- **[Backends](backends.md)** — Comparing pyahocorasick vs NumPy vs ONNX
- **[Examples](examples.md)** — Real-world usage patterns
- **[Troubleshooting](troubleshooting.md)** — Common issues
