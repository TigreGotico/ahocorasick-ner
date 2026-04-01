
# FAQ - Ahocorasick NER

### Q: Why Aho-Corasick?
**A:** Standard regex matching becomes significantly slower as the number of patterns increases. Aho-Corasick constructs a finite state machine once, allowing for simultaneous matching of any number of patterns in a single pass over the input text.

### Q: Does this handle fuzzy matching?
**A:** No. Aho-Corasick is a literal matching algorithm. If you need fuzzy matching, consider `rapidfuzz` or similar libraries. However, for large vocabularies (thousands of entries), literal matching is often much more practical.

### Q: How are overlapping entities handled?
**A:** This implementation uses a greedy approach: the longest matching entity is selected. If two entities start at the same position, the longer one wins. If they overlap, the first one encountered (which is also the longest due to sorting) wins.

### Q: How do I use this with OpenVoiceOS?
**A:** This package includes an `IntentTransformer` plugin. Once installed, it automatically listens for entity registration events (`padatious:register_entity`) and uses them to perform NER on incoming utterances.

### Q: Can I save/load my models?
**A:** Yes, the `AhocorasickNER` class has `save` and `load` methods that use `pickle` to serialize the underlying `pyahocorasick` automaton.

### Q: How is vocabulary RAM estimated?
**A:** The estimate assumes ~64 bytes per trie node per entity. This is a rough heuristic; actual memory depends on Aho-Corasick automaton structure and your system's Python runtime.
