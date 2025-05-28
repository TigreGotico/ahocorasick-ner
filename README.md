[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TigreGotico/ahocorasick-ner)

# AhocorasickNER

A fast and simple Named Entity Recognition (NER) tool based on the Aho-Corasick algorithm. This package is ideal for rule-based entity extraction using pre-defined vocabularies, especially when speed and scalability matter.

---

## ✨ Features

- ✅ Ultra-fast multi-pattern string matching using [Aho-Corasick](https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm)
- ✅ Word-boundary-aware matching
- ✅ Case-sensitive or case-insensitive modes
- ✅ Minimal dependencies
- ✅ Designed for integration with Hugging Face Datasets and similar sources

---

## 🧠 Theoretical Background

The Aho-Corasick algorithm, developed by Alfred V. Aho and Margaret J. Corasick in 1975, constructs a **finite state machine** from a set of keywords to allow **simultaneous pattern matching** in linear time. It's similar to a Trie, but extended with failure transitions that allow it to efficiently handle mismatches and overlapping substrings.

This approach is **ideal for dictionary-based NER systems**, where:
- You have a fixed list of entities (e.g., names, locations, products)
- You want to search for **many patterns in a single pass**
- Speed and low memory usage are critical

Unlike statistical or neural NER systems, this approach doesn't require training and is fully deterministic. It is particularly useful when:
- You want **consistent results**
- You are working with **domain-specific vocabularies**
- You need to process **large corpora quickly**

---

## 🚀 Installation

```bash
pip install ahocorasick-ner
```

---

## 🛠️ Usage

```python
from ahocorasick_ner import AhocorasickNER
from datasets import load_dataset

EncyclopediaMetallvm = AhocorasickNER()

dataset_name = "Jarbas/metal-archives-tracks"
dataset = load_dataset(dataset_name)["train"]
for entry in dataset:
    EncyclopediaMetallvm.add_word("artist_name", entry["band_name"])
    if entry.get("track_name"):
        EncyclopediaMetallvm.add_word("track_name", entry["track_name"])
    if entry.get("album_name"):
        EncyclopediaMetallvm.add_word("album_name", entry["album_name"])
    EncyclopediaMetallvm.add_word("album_type", entry["album_type"])

dataset_name = "Jarbas/metal-archives-bands"
dataset = load_dataset(dataset_name)["train"]
for entry in dataset:
    EncyclopediaMetallvm.add_word("artist_name", entry["name"])
    if entry.get("genre"):
        EncyclopediaMetallvm.add_word("music_genre", entry["genre"])
    if entry.get("label"):
        EncyclopediaMetallvm.add_word("record_label", entry["label"])
    if entry.get("country"):
        EncyclopediaMetallvm.add_word("country", entry["country"])

for entity in EncyclopediaMetallvm.tag("I fucking love black metal from Norway"):
    print(entity)
```

### Output:
```python
{'start': 15, 'end': 25, 'word': 'black metal', 'label': 'genre'}
{'start': 32, 'end': 37, 'word': 'Norway', 'label': 'country'}
```

---

## 🧪 Benchmarks

With 100k+ known phrases, this tool can tag documents in milliseconds thanks to the Aho-Corasick FSM structure. It scales gracefully with both the number of patterns and the size of input text.

---

## 🧩 Limitations

- Does not handle nested or overlapping entities well (greedy, longest match wins)
- No fuzzy matching (e.g., typos or misspellings won't match)
- Requires all entities to be known beforehand


---

## 📄 License

MIT — free for commercial and non-commercial use.

---

## 🙏 Acknowledgements

- [pyahocorasick](https://github.com/WojciechMula/pyahocorasick) — The underlying C-based Aho-Corasick implementation
- [Hugging Face Datasets](https://huggingface.co/docs/datasets) — For loading domain-specific corpora
