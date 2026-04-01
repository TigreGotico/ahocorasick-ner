"""Basic usage of AhocorasickNER with the pyahocorasick backend."""
from ahocorasick_ner import AhocorasickNER

ner = AhocorasickNER()

# Add city names
cities = [
    ("city", "New York"),
    ("city", "Los Angeles"),
    ("city", "San Francisco"),
    ("city", "London"),
    ("city", "Paris"),
    ("city", "Tokyo"),
    ("city", "Berlin"),
    ("city", "Moscow"),
    ("city", "Sydney"),
    ("city", "Buenos Aires"),
]
for label, name in cities:
    ner.add_word(label, name)

# Add country names
countries = [
    ("country", "United States"),
    ("country", "United Kingdom"),
    ("country", "France"),
    ("country", "Japan"),
    ("country", "Germany"),
    ("country", "Russia"),
    ("country", "Australia"),
    ("country", "Argentina"),
]
for label, name in countries:
    ner.add_word(label, name)

ner.fit()

# Tag some text
sentences = [
    "I traveled from New York to London last summer.",
    "The conference in Tokyo was about AI research in Japan.",
    "She moved from Buenos Aires to San Francisco for work.",
    "Paris and Berlin are beautiful European cities.",
]

for sentence in sentences:
    print(f"\n> {sentence}")
    for entity in ner.tag(sentence):
        print(f"  {entity['label']:>10}: {entity['word']}")
