# Examples

Real-world usage patterns and complete working examples.

---

## Quick Examples

### Example 1: Music Recommendation

Extract artist and album names from user input.

```python
from ahocorasick_ner import AhocorasickNER

# Train
ner = AhocorasickNER(case_sensitive=False)
ner.add_word("artist", "Metallica")
ner.add_word("artist", "Iron Maiden")
ner.add_word("artist", "Black Sabbath")
ner.add_word("album", "Master of Puppets")
ner.add_word("album", "The Number of the Beast")
ner.add_word("album", "Paranoid")
ner.fit()

# Use
user_input = "I love Metallica's Master of Puppets and Iron Maiden"
entities = list(ner.tag(user_input))

for entity in entities:
    if entity["label"] == "artist":
        print(f"🎤 Artist: {entity['word']}")
    elif entity["label"] == "album":
        print(f"💿 Album: {entity['word']}")

# Output:
# 🎤 Artist: Metallica
# 💿 Album: Master of Puppets
# 🎤 Artist: Iron Maiden
```

---

### Example 2: Product Tagging

Identify brands and product types in e-commerce descriptions.

```python
from ahocorasick_ner import AhocorasickNER

ner = AhocorasickNER()

# Brands
for brand in ["Nike", "Adidas", "Puma", "Reebok"]:
    ner.add_word("brand", brand)

# Product types
for product in ["running shoe", "basketball shoe", "sneaker", "cleats"]:
    ner.add_word("product_type", product)

ner.fit()

# Process product descriptions
descriptions = [
    "Nike running shoes with premium cushioning",
    "Adidas basketball shoe for professional athletes",
    "Puma cleats for soccer"
]

for desc in descriptions:
    print(f"\n{desc}")
    entities = list(ner.tag(desc))
    for entity in entities:
        print(f"  [{entity['label']}] {entity['word']}")
```

---

### Example 3: Content Filtering

Detect blacklisted terms or flagged content.

```python
from ahocorasick_ner import AhocorasickNER

# Blocklist
ner = AhocorasickNER()
for blocked in ["spam", "scam", "malware", "phishing"]:
    ner.add_word("blocklisted", blocked)

ner.fit()

def is_safe(text):
    matches = list(ner.tag(text))
    if matches:
        print(f"⚠️  Blocked terms found:")
        for m in matches:
            print(f"   - {m['word']}")
        return False
    return True

print("Test 1:", is_safe("Check out this amazing offer!"))
print("Test 2:", is_safe("Click this link for a spam opportunity"))
```

---

## Batch Processing

Process multiple documents efficiently.

### Pattern: Iterate Over Documents

```python
from ahocorasick_ner import AhocorasickNER

ner = AhocorasickNER()
# ... add words and fit ...

documents = [
    "Metallica released Master of Puppets in 1986",
    "Iron Maiden formed in 1975",
    "Black Sabbath pioneered heavy metal"
]

results = []
for doc_id, doc_text in enumerate(documents):
    entities = list(ner.tag(doc_text))
    for entity in entities:
        results.append({
            "doc_id": doc_id,
            "text": doc_text[:50],
            "entity": entity["word"],
            "label": entity["label"],
            "position": f"{entity['start']}-{entity['end']}"
        })

# Export to CSV
import csv
with open("entities.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["doc_id", "text", "entity", "label", "position"])
    writer.writeheader()
    writer.writerows(results)
```

### Pattern: Parallel Processing

```python
from ahocorasick_ner import AhocorasickNER
from concurrent.futures import ThreadPoolExecutor
import time

# Create NER system (thread-safe)
ner = AhocorasickNER()
for i in range(1000):
    ner.add_word(f"entity_{i % 10}", f"term_{i}")
ner.fit()

# Process documents in parallel
documents = [f"term_{i % 100} is interesting" for i in range(10000)]

def process_doc(doc):
    return list(ner.tag(doc))

start = time.time()
with ThreadPoolExecutor(max_workers=4) as executor:
    all_results = list(executor.map(process_doc, documents))
elapsed = time.time() - start

print(f"Processed {len(documents)} docs in {elapsed:.2f}s")
print(f"Found {sum(len(r) for r in all_results)} total entities")
```

---

## Persistence

Save and load trained models.

### Save for Reuse

```python
from ahocorasick_ner import AhocorasickNER

# Train (one time, expensive)
ner = AhocorasickNER()
for i in range(100000):
    ner.add_word(f"entity_{i % 1000}", f"term_{i}")
ner.fit()

# Save to disk
ner.save("large_vocabulary.ahocorasick")
print("✅ Saved 100K entities")
```

### Load and Use

```python
from ahocorasick_ner import AhocorasickNER

# Load (one time, fast)
ner = AhocorasickNER()
ner.load("large_vocabulary.ahocorasick")

# Use immediately (no re-fitting)
for text in ["term_5000 is here", "term_9999 is there"]:
    entities = list(ner.tag(text))
    print(f"{text} -> {len(entities)} entities found")
```

---

## Integration: API Endpoint

Use ahocorasick-ner in a REST API.

```python
from ahocorasick_ner import AhocorasickNER
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize NER (load once)
ner = AhocorasickNER()
ner.load("pretrained_model.ahocorasick")

app = FastAPI()

class TextInput(BaseModel):
    text: str
    min_word_len: int = 5

class Entity(BaseModel):
    word: str
    label: str
    start: int
    end: int

class TagResponse(BaseModel):
    text: str
    entities: list[Entity]

@app.post("/tag")
async def tag_text(request: TextInput) -> TagResponse:
    entities = [
        Entity(**entity)
        for entity in ner.tag(request.text, min_word_len=request.min_word_len)
    ]
    return TagResponse(text=request.text, entities=entities)

# Usage:
# curl -X POST http://localhost:8000/tag \
#   -H "Content-Type: application/json" \
#   -d '{"text": "Metallica rocks!", "min_word_len": 5}'
```

---

## Integration: Database

Store extracted entities in a database.

```python
from ahocorasick_ner import AhocorasickNER
import sqlite3

# Setup
ner = AhocorasickNER()
ner.load("pretrained_model.ahocorasick")

db = sqlite3.connect("entities.db")
cursor = db.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS extracted_entities (
    id INTEGER PRIMARY KEY,
    document_id TEXT,
    entity_text TEXT,
    label TEXT,
    start_pos INTEGER,
    end_pos INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Process and store
documents = {
    "doc_001": "Metallica performed at Wembley Stadium",
    "doc_002": "Iron Maiden's album The Number of the Beast",
}

for doc_id, text in documents.items():
    for entity in ner.tag(text):
        cursor.execute(
            "INSERT INTO extracted_entities VALUES (NULL, ?, ?, ?, ?, ?, DEFAULT)",
            (doc_id, entity["word"], entity["label"], entity["start"], entity["end"])
        )

db.commit()

# Query results
cursor.execute("SELECT * FROM extracted_entities WHERE label = 'band'")
for row in cursor.fetchall():
    print(row)

db.close()
```

---

## Advanced: Dynamic Entity Management

Add/remove entities without re-training entire system.

```python
from ahocorasick_ner import AhocorasickNER

class DynamicNER:
    def __init__(self):
        self.ner = AhocorasickNER()
        self.entities = {}  # Track for management

    def add_entity(self, label, text):
        """Add entity and re-fit"""
        self.ner.add_word(label, text)
        if label not in self.entities:
            self.entities[label] = []
        self.entities[label].append(text)
        self.ner.fit()

    def tag(self, text):
        """Extract entities"""
        return list(self.ner.tag(text))

    def list_entities(self, label=None):
        """List all registered entities"""
        if label:
            return self.entities.get(label, [])
        return self.entities

    def remove_entity(self, label, text):
        """Remove entity and re-train from scratch"""
        if label in self.entities and text in self.entities[label]:
            self.entities[label].remove(text)

            # Rebuild automaton from scratch
            self.ner = AhocorasickNER()
            for lbl, texts in self.entities.items():
                for txt in texts:
                    self.ner.add_word(lbl, txt)
            self.ner.fit()

# Usage
dnr = DynamicNER()
dnr.add_entity("band", "Metallica")
dnr.add_entity("band", "Iron Maiden")
print(dnr.tag("Metallica rocks"))

dnr.remove_entity("band", "Metallica")
print(dnr.tag("Metallica rocks"))  # No match now

print(dnr.list_entities())
```

---

## Preprocessing: Text Normalization

Clean text before tagging.

```python
from ahocorasick_ner import AhocorasickNER
import re
import unicodedata

def normalize_text(text):
    """Clean and normalize text for NER"""
    # Remove diacritics (é -> e)
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    # Replace unicode quotes with ASCII
    text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

ner = AhocorasickNER()
ner.add_word("artist", "Jose Maria")
ner.fit()

# Raw text with accents
raw = "I like José María"
normalized = normalize_text(raw)

print(f"Raw: {raw}")
print(f"Normalized: {normalized}")
print(f"Entities: {list(ner.tag(normalized))}")
```

---

## Performance: Bulk Tagging

Optimize for processing many documents.

```python
from ahocorasick_ner import AhocorasickNER
import time

# Train large system
ner = AhocorasickNER()
for i in range(50000):
    ner.add_word(f"entity_{i % 100}", f"word_{i}")
ner.fit()

# Method 1: Direct tagging (simple)
texts = ["word_500 is here"] * 10000
start = time.time()
results1 = [list(ner.tag(t)) for t in texts]
time1 = time.time() - start

print(f"Method 1 (direct): {time1:.2f}s")

# Method 2: Generator (memory-efficient)
def process_texts_generator(ner, texts):
    for text in texts:
        yield list(ner.tag(text))

start = time.time()
results2 = list(process_texts_generator(ner, texts))
time2 = time.time() - start

print(f"Method 2 (generator): {time2:.2f}s")

# Method 3: Parallel (multiple workers)
from concurrent.futures import ThreadPoolExecutor

start = time.time()
with ThreadPoolExecutor(max_workers=4) as exe:
    results3 = list(exe.map(lambda t: list(ner.tag(t)), texts))
time3 = time.time() - start

print(f"Method 3 (parallel): {time3:.2f}s")
```

---

## See Also

- **[API Reference](api-reference.md)** — Full method documentation
- **[Datasets](datasets.md)** — Using pre-built vocabularies
- **[Integration](integration.md)** — OVOS plugin integration
- **[Performance](performance.md)** — Benchmarks and optimization
