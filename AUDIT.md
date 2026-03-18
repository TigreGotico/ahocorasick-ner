
# ahocorasick-ner — Audit Report

## Documentation Status
- [x] FAQ.md — Q&A for core NER engine
- [x] MAINTENANCE_REPORT.md — change history
- [x] AUDIT.md — this file
- [x] docs/index.md — comprehensive architecture & usage guide

## Implementation Status

### Core NER Engine
- [x] Aho-Corasick automaton wrapper — `AhocorasickNER` class — `ahocorasick_ner/__init__.py:7`
- [x] Word boundary detection — `tag()` method respects regex word boundaries
- [x] Overlap resolution — greedy longest-match strategy — `ahocorasick_ner/__init__.py:96`
- [x] Save/load via pickle — `save()` and `load()` methods

### Dataset Loaders
- [x] Metal Archives — `EncyclopediaMetallvmNER` — `ahocorasick_ner/datasets.py:14`
- [x] Multi-genre music — `MusicNER` — `ahocorasick_ner/datasets.py:69`
- [x] IMDB entities — `ImdbNER` — `ahocorasick_ner/datasets.py:167`

## Known Limitations & Design Notes

### RAM Estimation
- Estimate assumes ~64 bytes/trie node (heuristic)
- Actual memory: system-dependent (Python runtime + automaton structure)
- Latency estimate: single benchmark pass (not production profile)
