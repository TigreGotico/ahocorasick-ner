# Troubleshooting

Common issues and solutions.

---

## Installation Issues

### ImportError: No module named 'ahocorasick'

**Cause:** Library not installed or wrong environment

**Solution:**
```bash
uv pip install ahocorasick-ner
```text

Or activate correct virtual environment:
```bash
source ~/.venvs/my_env/bin/activate
pip install ahocorasick-ner
```text

---

### error: Microsoft Visual C++ 14.0 is required (Windows)

**Cause:** C compiler missing (pyahocorasick requires compilation)

**Solution:**

1. **Install Visual C++ Build Tools:**
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Run installer, select "Desktop development with C++"

2. **Or use NumPy backend (no compilation):**
   ```bash
   pip install ahocorasick-ner[numpy]
```text

3. **Or use pre-compiled wheel (if available):**
   ```bash
   pip install --only-binary :all: ahocorasick-ner
```text

---

### ImportError: cannot import name 'load_dataset'

**Cause:** HuggingFace `datasets` library not installed

**Solution:**
```bash
uv pip install ahocorasick-ner[datasets]
```text

Or install separately:
```bash
pip install datasets
```text

---

## Usage Issues

### No matches found (entities not extracted)

**Cause 1: Automaton not fitted**

```python
ner = AhocorasickNER()
ner.add_word("artist", "Metallica")
# ❌ Forgot to call fit()
list(ner.tag("I like Metallica"))  # Returns empty

# ✅ Fix:
ner.fit()
list(ner.tag("I like Metallica"))  # Now returns matches
```text

**Cause 2: Text doesn't match exactly**

```python
ner = AhocorasickNER(case_sensitive=False)
ner.add_word("artist", "Metallica")
ner.fit()

# ❌ No match (default case-insensitive requires lowercase)
list(ner.tag("I LIKE metallica"))  # Matches "metallica"

# But case is preserved in output
list(ner.tag("I LIKE MetallicA"))  # Returns {'word': 'MetallicA', ...}
```text

**Cause 3: Word boundaries blocking match**

```python
ner = AhocorasickNER()
ner.add_word("word", "iron")
ner.fit()

# ❌ No match (underscore is word character)
list(ner.tag("This_iron_will"))  # Empty

# ✅ Matches with space or punctuation
list(ner.tag("This iron-will"))  # [match]
list(ner.tag("This iron."))      # [match]
```text

**Debug:**
```python
def debug_tag(ner, text):
    # Enable detailed logging
    for i, char in enumerate(text):
        if char.isalnum():
            print(f"  [{i}] {char} - word character")
        else:
            print(f"  [{i}] {repr(char)} - non-word")

debug_tag(ner, "This_iron_will")
```text

---

### Overlapping entities all selected (want longest only)

**Issue:** Multiple overlapping matches returned, but only longest should be selected

**Cause:** Bug in custom overlap code (shouldn't happen with built-in `tag()`)

**Solution:** Verify using built-in `tag()` method:

```python
# ✅ Correct: Built-in tag() handles overlaps
ner.add_word("entity", "abc")
ner.add_word("entity", "bcd")
ner.add_word("entity", "bcde")
ner.fit()

matches = list(ner.tag("abcde"))
# [{'start': 1, 'end': 4, 'word': 'bcde', 'label': 'entity'}]
# Only "bcde" returned (longest)
```text

---

### Case sensitivity not working

**Issue: Case-insensitive matching when case-sensitive expected**

```python
# ❌ Expected case-sensitive but got case-insensitive
ner = AhocorasickNER(case_sensitive=False)  # ← Default is False!
ner.add_word("artist", "Metallica")
ner.fit()

list(ner.tag("METALLICA"))  # [match] - case-insensitive
```text

**Solution:**
```python
# ✅ Enable case sensitivity
ner = AhocorasickNER(case_sensitive=True)
ner.add_word("artist", "Metallica")  # Exact case required
ner.fit()

list(ner.tag("METALLICA"))   # [] - no match
list(ner.tag("Metallica"))   # [match]
```text

---

### Performance degradation with many entities

**Cause 1: Re-fitting on every use**

```python
# ❌ Slow: Re-fit every request
def handle_request(text):
    ner = AhocorasickNER()
    ner.add_word("artist", "Metallica")
    ner.fit()  # 50ms every request!
    return list(ner.tag(text))
```text

**Solution: Load once, reuse**

```python
# ✅ Fast: Load once at startup
ner = AhocorasickNER()
ner.load("prebuilt_model.ahocorasick")

def handle_request(text):
    return list(ner.tag(text))  # <5ms
```text

**Cause 2: Very large vocabulary**

```python
# ❌ Slow: 100K entities
ner = AhocorasickNER()
for i in range(100000):
    ner.add_word("entity", f"term_{i}")
ner.fit()  # ~600ms
```text

**Solution: Reduce vocabulary or split into categories**

```python
# ✅ Split by category
ner_artist = AhocorasickNER()
for artist in artists:  # 10K entities
    ner_artist.add_word("artist", artist)
ner_artist.fit()

ner_album = AhocorasickNER()
for album in albums:  # 10K entities
    ner_album.add_word("album", album)
ner_album.fit()
```text

---

## File I/O Issues

### FileNotFoundError when loading

**Cause:** Model file doesn't exist or wrong path

```python
# ❌ Wrong path
ner.load("my_model.ahocorasick")  # File not found
```text

**Solution:**
```python
import os

# ✅ Check path exists
model_path = "my_model.ahocorasick"
if not os.path.exists(model_path):
    # Train and save
    ner = AhocorasickNER()
    ner.add_word("artist", "Metallica")
    ner.fit()
    ner.save(model_path)
else:
    # Load existing
    ner = AhocorasickNER()
    ner.load(model_path)
```text

---

### File corrupted when loading

**Cause:** Model file corrupted or from different Python version

**Solution:**
```python
# ✅ Rebuild if load fails
try:
    ner.load("my_model.ahocorasick")
except (EOFError, pickle.UnpicklingError):
    # Corrupted file, rebuild
    ner = AhocorasickNER()
    ner.add_word("artist", "Metallica")
    ner.fit()
    ner.save("my_model.ahocorasick")
```text

---

## Backend-Specific Issues

### NumPy backend slower than expected

**Cause:** NumPy not optimized for your use case

**Solution:**
```python
# ✅ Switch to pyahocorasick for performance
from ahocorasick_ner import AhocorasickNER
ner = AhocorasickNER()  # C-based, faster

# Or optimize NumPy usage:
from ahocorasick_ner.numpy_backend import NumpyAhocorasickNER
ner = NumpyAhocorasickNER()
# Use batch processing to amortize overhead
```text

---

### ONNX model too large

**Cause:** Serialization overhead

**Solution:**
```python
# ✅ Use pyahocorasick for smaller files
from ahocorasick_ner import AhocorasickNER
ner = AhocorasickNER()

# ✅ Reduce vocabulary size
# Remove unnecessary entities before saving
```text

---

## OpenVoiceOS Integration Issues

### Entities not recognized in OVOS skill

**Cause 1: Plugin not loaded**

```bash
# Check if plugin is available
ovos-config show | grep ahocorasick
```text

**Solution:**
```bash
# Reinstall
uv pip install ahocorasick-ner
```text

**Cause 2: Entities not registered**

```python
# ❌ Forgot to register
class MySkill(OVOSSkill):
    def initialize(self):
        pass  # No entity registration

# ✅ Register entities
class MySkill(OVOSSkill):
    def initialize(self):
        self.register_entity("artist", ["Metallica"])
```text

**Cause 3: Text doesn't match exactly**

```python
def handle_music(self, message):
    entities = message.data.get("entities", [])
    # If text is "play metallica" (lowercase)
    # But registered as "Metallica" (capitalized)
    # No match (case-sensitive by default)
```text

**Solution:**
```python
# Use case-insensitive matching
self.register_entity("artist", ["metallica"])  # lowercase
```text

---

## Memory Issues

### Out of memory with large vocabulary

**Cause:** Loading huge dataset into memory

```python
# ❌ Out of memory: 100K entities
from ahocorasick_ner.datasets import MusicNER
ner = MusicNER()  # Takes ~150MB
```text

**Solution:**
```python
# ✅ Use smaller dataset
from ahocorasick_ner.datasets import EncyclopediaMetallvmNER
ner = EncyclopediaMetallvmNER()  # Takes ~50MB

# ✅ Or reduce vocabulary in custom NER
ner = AhocorasickNER()
# Add only necessary entities (not all 100K)
for entity in necessary_entities:
    ner.add_word("entity", entity)
ner.fit()
```text

---

## Thread Safety

### Issues with multithreading

**Cause:** Modifying NER while tagging in another thread

```python
# ❌ Not thread-safe: modify while tagging
ner = AhocorasickNER()
ner.add_word("artist", "Metallica")
ner.fit()

# Thread 1: Tagging
for t in texts:
    list(ner.tag(t))

# Thread 2: Adding entities (BAD!)
ner.add_word("artist", "Iron Maiden")  # Race condition
```text

**Solution: Load once, use read-only**

```python
# ✅ Thread-safe: load once, tag from multiple threads
ner = AhocorasickNER()
ner.load("prebuilt_model.ahocorasick")

from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as exe:
    # Multiple threads tag safely
    results = list(exe.map(lambda t: list(ner.tag(t)), texts))
```text

---

## Getting Help

### Check Logs

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ahocorasick_ner")
logger.setLevel(logging.DEBUG)

ner = AhocorasickNER()
ner.add_word("artist", "Metallica")
ner.fit()
list(ner.tag("Metallica"))
```text

### Inspect Internal State

```python
# Check fitted status
print(f"Fitted: {ner._fitted}")

# Check entities in automaton
print(f"Automaton: {ner.automaton}")

# Check case sensitivity
print(f"Case sensitive: {ner.case_sensitive}")
```text

### Run Tests

```bash
uv run pytest test/unittests/ -v
```text

If tests pass but your code fails, isolate the issue:

```python
# Test basic functionality
from ahocorasick_ner import AhocorasickNER

ner = AhocorasickNER()
ner.add_word("label", "text")
ner.fit()
results = list(ner.tag("text"))
assert len(results) == 1
assert results[0]["word"] == "text"
print("✓ Basic test passed")
```text

---

## See Also

- **[API Reference](api-reference.md)** — Method signatures
- **[Examples](examples.md)** — Working code samples
- **[Performance](performance.md)** — Optimization
- **[Algorithms](algorithms.md)** — How it works
