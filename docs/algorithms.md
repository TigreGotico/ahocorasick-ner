# Algorithms

Deep dive into the Aho-Corasick algorithm, word boundaries, and overlap resolution.

---

## The Aho-Corasick Algorithm

A finite-state machine for simultaneous multi-pattern string matching.

### Why Aho-Corasick?

**Naive approach** (search for each pattern separately):
```python
entities = ["Metallica", "Iron Maiden", "Black Sabbath"]
text = "I like Metallica, Iron Maiden, and Black Sabbath"

for entity in entities:
    if entity in text:
        print(f"Found: {entity}")
# Time: O(n * m * k)  where n=patterns, m=text length, k=pattern length
```text

**Aho-Corasick approach** (single pass):
```python
# Build FSM once: O(m) where m = sum of pattern lengths
# Search all patterns: O(n) where n = text length
# Total: O(m + n)  regardless of number of patterns!
```text

### How It Works

#### 1. Build a Trie (Prefix Tree)

Given entities: "he", "she", "his", "hers"

```text
       root
      / | \
     h  s  [others]
    / \ |
   e   i s
  / \ |
 (he) s h
     / | \
    e  (his)
    |
    r
    |
    s
    |
   (hers)
```text

**Complexity:** O(m) where m = sum of all pattern lengths

#### 2. Add Failure Links (BFS)

When a character doesn't match, jump to the longest suffix that *is* a prefix of another pattern.

```text
Input: "ushers"
       u -> [fail] -> s
       s -> h
       h -> e
       e -> r
       r -> s (match "hers")
```text

**Complexity:** O(m + alphabet_size)

#### 3. Search with FSM

Traverse the text character-by-character, following edges or failure links:

```python
# Pseudocode
current_state = root
for char in text:
    while current_state and char not in current_state.edges:
        current_state = current_state.fail_link
    current_state = current_state.edges.get(char, root)
    if current_state.is_pattern:
        yield match(current_state.pattern, position)
```text

**Complexity:** O(n + z) where n = text length, z = matches

---

## Word Boundaries

By default, matches must be separated from alphanumeric characters.

### The Problem

Without word boundaries:
```python
ner = AhocorasickNER()
ner.add_word("artist", "Iron")
ner.fit()

list(ner.tag("We admire iron-will and iron gates"))
# Matches "iron" in: "admire iron-will" (WRONG — inside "iron-will")
```text

### The Solution

Check characters before and after the match — `ahocorasick_ner/__init__.py:88-92`:

```python
before = processed_haystack[start - 1] if start > 0 else ' '
after = processed_haystack[end + 1] if end + 1 < len(processed_haystack) else ' '
if re.match(r'\w', before) or re.match(r'\w', after):
    continue  # skip: word character before/after = partial match
```text

The regex `\w` matches: `[a-zA-Z0-9_]`

### Example

```python
text = "Iron is iron (element), Iron Maiden (band)"
       # 0   1   23   4567   890123    45678 9

ner = AhocorasickNER()
ner.add_word("band", "Iron")  # Exact match only
ner.add_word("element", "iron")
ner.fit()

matches = list(ner.tag(text))
# [{'start': 14, 'end': 17, 'word': 'iron', 'label': 'element'},
#  {'start': 33, 'end': 36, 'word': 'Iron', 'label': 'band'}]

# Skipped: "Iron" at start (capital, matches lowercase "iron" label)
```text

### Edge Cases

**Underscores are word characters:**
```python
ner = AhocorasickNER()
ner.add_word("var", "foo")
ner.fit()

list(ner.tag("my_foo_bar"))  # [] — no match (underscore blocks it)
list(ner.tag("my-foo-bar"))  # [match] — hyphen doesn't block it
```text

**Customize via min_word_len:**

`min_word_len` controls the minimum match *length*; word-boundary checks always run regardless of this value.

```python
# Allow short matches by reducing min_word_len (boundary checks still apply)
ner.add_word("word", "a")  # 1 character
ner.fit()

list(ner.tag("a apple", min_word_len=1))  # "a" at position 0 matches (length >= 1, has boundaries)
```text

---

## Overlap Resolution

When multiple entities overlap, select non-overlapping matches using greedy longest-match-first.

### The Problem

Given entities: "abc", "bcd", "bcde"
Text: "abcde"

Possible matches:
- "abc" at (0, 2)
- "bcd" at (1, 3)
- "bcde" at (1, 4)

Which to select?

### The Algorithm

**Step 1: Sort by length (descending), then by start position** — `ahocorasick_ner/__init__.py:97`:

```python
matches = [(0, 2, "abc"), (1, 3, "bcd"), (1, 4, "bcde")]
sorted_matches = [
    (1, 4, "bcde"),   # Longest: 4 chars
    (1, 3, "bcd"),    # Middle: 3 chars
    (0, 2, "abc"),    # Shortest: 3 chars (tie, but starts earlier)
]
```text

**Step 2: Greedy selection** — `ahocorasick_ner/__init__.py:99-105`:

```python
selected = []
used_positions = set()

for start, end, word, label in sorted_matches:
    # Check if any position in [start, end] already used
    if all(i not in used_positions for i in range(start, end + 1)):
        selected.append((start, end, word, label))
        used_positions.update(range(start, end + 1))
```text

**Result:**
```text
Positions: 0 1 2 3 4
Match 1:   [b c d e]     — selected (no conflicts)
Match 2:       [b c d]   — skipped (overlaps with match 1)
Match 3: [a b c]         — skipped (overlaps with match 1)

Selected: [(1, 4, "bcde")]
```text

### Examples

**Example 1: Multiple overlaps**
```python
ner = AhocorasickNER()
ner.add_word("entity", "the")
ner.add_word("entity", "cat")
ner.add_word("entity", "the cat")
ner.fit()

text = "I see the cat"
matches = list(ner.tag(text))
# [{'start': 7, 'end': 13, 'word': 'the cat', 'label': 'entity'}]
# Selected: "the cat" (longest, 7 chars)
# Skipped: "the" and "cat" (subsumed by longer match)
```text

**Example 2: Non-overlapping matches**
```python
ner = AhocorasickNER()
ner.add_word("band", "Metallica")
ner.add_word("band", "Iron Maiden")
ner.add_word("album", "Master of Puppets")
ner.fit()

text = "Metallica's Master of Puppets and Iron Maiden's The Number of the Beast"
matches = list(ner.tag(text))
# All four matches selected (no overlaps)
# [{'start': 0, ...}, {'start': 13, ...}, {'start': 48, ...}, {'start': 62, ...}]
```text

**Example 3: Same length, different start positions**
```python
ner = AhocorasickNER()
ner.add_word("entity", "ab")
ner.add_word("entity", "bc")
ner.fit()

text = "abc"
matches = list(ner.tag(text, min_word_len=1))
# Sorted: [(0, 1, "ab"), (1, 2, "bc")]  — same length, sorted by start
# Selected: (0, 1, "ab")  — chosen first, blocks (1, 2)
```text

---

## Case Sensitivity

Controlled by the `case_sensitive` flag in constructor.

### Case-Insensitive (Default)

```python
ner = AhocorasickNER(case_sensitive=False)
ner.add_word("band", "Metallica")
ner.fit()

matches = list(ner.tag("I LOVE METALLICA"))
# [{'word': 'METALLICA', 'label': 'band'}]
# Original case preserved, but match is case-insensitive
```text

**Implementation** — `ahocorasick_ner/__init__.py:51, 78`:
```python
# During add_word:
key = example.lower() if not self.case_sensitive else example

# During tag:
processed_haystack = haystack.lower() if not self.case_sensitive else haystack
# ...search in lowercase...
# But yield original case from haystack
```text

### Case-Sensitive

```python
ner = AhocorasickNER(case_sensitive=True)
ner.add_word("band", "Metallica")  # Exact case
ner.fit()

list(ner.tag("I LOVE METALLICA"))       # [] — no match (wrong case)
list(ner.tag("I LOVE Metallica"))       # [match]
list(ner.tag("I LOVE metallica"))       # [] — no match
```text

---

## Complexity Analysis

### Build Phase (Trie + Failure Links)

```text
Operation         | Complexity | Notes
------------------|-----------|---------------------------------------------
Create empty trie | O(1)       | Just root node
Add N entities    | O(m)       | m = sum of entity lengths
Build FSM         | O(m)       | BFS to compute failure links
Total             | O(m)       | Practical: 50ms for 10K entities
```text

### Search Phase (Match)

```text
Operation         | Complexity | Notes
------------------|-----------|---------------------------------------------
Scan text         | O(n)       | n = text length
Follow edges/fails| O(1)       | Per character (amortized)
Report matches    | O(z)       | z = number of matches
Total             | O(n + z)   | Practical: 5ms for 10K chars
```text

### Memory

```text
Component         | Usage     | Notes
------------------|-----------|---------------------------------------------
Trie structure    | O(m)      | m = sum of entity lengths
Failure links     | O(m)      | One per trie node
Automaton cache   | O(m)      | pyahocorasick uses state compression
Total per entity  | ~64 bytes | Heuristic estimate per trie node
```text

### Comparison to Alternatives

```text
Algorithm         | Build | Search | Setup | Memory | Typos
------------------|-------|--------|-------|--------|-------
Aho-Corasick      | O(m)  | O(n+z) | Easy  | Low    | None
Regex (compiled)  | O(m)  | O(n*m) | Easy  | Low    | Hard
Regex (naive)     | O(1)  | O(n*m) | Fast  | Low    | Hard
Fuzzy match       | O(m)  | O(n*m) | Hard  | High   | Yes
Transformer NER   | O(1)  | O(n)   | Very  | Very   | Yes
              |      |        | Hard  | High   |
```text

---

## Implementation Details

### Trie Node Structure

```python
# Conceptual (pyahocorasick uses optimized binary format)
class TrieNode:
    edges: Dict[str, TrieNode]        # Character -> next node
    fail_link: Optional[TrieNode]     # Failure link (BFS computed)
    is_pattern: bool                   # Does this node end a pattern?
    pattern: Optional[Tuple[str, str]] # (label, word) if is_pattern
```text

### Automaton State Representation

pyahocorasick uses an efficient binary format:

```text
State ID | Edges | Fail Link | Pattern? | Pattern Data
---------|-------|-----------|----------|-------------------
0        | 256   | -1        | No       | None
1        | 256   | 0         | Yes      | ("artist", "Iron")
2        | 256   | 1         | No       | None
...      | ...   | ...       | ...      | ...
```text

See `pyahocorasick` documentation for C-level details.

### Algorithm Properties

| Property | Value |
|----------|-------|
| **Deterministic** | Yes — same input always produces same output |
| **Online** | Yes — can process text as stream |
| **Stateful** | Yes — maintains current FSM state between characters |
| **Space-optimal** | No — trie can be trie-compressed (McCreight suffix links) |
| **Time-optimal** | Yes — provably O(n + z) for any DFA |

---

## See Also

- **[API Reference](api-reference.md)** — Using `tag()`, `fit()`, etc.
- **[Performance](performance.md)** — Profiling and benchmarks
- **[Backends](backends.md)** — Implementation differences
