
# QUICK_FACTS - Ahocorasick NER

| Detail | Information |
|--------|-------------|
| **Package Name** | `ahocorasick-ner` |
| **Version** | `0.1.2a2` (refer to `version.py`) |
| **Main Classes** | `AhocorasickNER`, `AhocorasickNERTransformer`, `MusicNER`, `ImdbNER` |
| **Dependencies** | `pyahocorasick`, `ovos-plugin-manager`, `ovos-utils`, `ovos-bus-client` |
| **Supported Python** | 3.10+ |
| **OPM Plugin** | `ovos-ahocorasick-ner-plugin` |
| **License** | Apache-2.0 |

## Key Methods

| Class | Method | Purpose |
|-------|--------|---------|
| `AhocorasickNER` | `add_word(label, example)` | Add an entity example to the matching dictionary. |
| `AhocorasickNER` | `tag(text)` | Search for entities in a string and return their positions and labels. |
| `AhocorasickNERTransformer` | `transform(intent)` | OPM integration to inject entities into intent matches. |
