"""Using the pure-numpy backend (no C extensions required at inference)."""
from ahocorasick_ner.numpy_backend import NumpyAhocorasickNER

ner = NumpyAhocorasickNER()

# Build a location recognizer
locations = {
    "city": [
        "New York", "Los Angeles", "San Francisco", "Chicago",
        "London", "Paris", "Tokyo", "Berlin", "Sydney", "Toronto",
    ],
    "country": [
        "United States", "United Kingdom", "France",
        "Japan", "Germany", "Australia", "Canada",
    ],
}

for label, names in locations.items():
    for name in names:
        ner.add_word(label, name)

ner.fit()

# Save and reload (uses .npz format)
ner.save("/tmp/locations.npz")
print("Saved to /tmp/locations.npz")

loaded = NumpyAhocorasickNER()
loaded.load("/tmp/locations.npz")
print("Loaded from /tmp/locations.npz")

text = "I flew from Tokyo to San Francisco via Los Angeles."
print(f"\n> {text}")
for entity in loaded.tag(text):
    print(f"  {entity['label']:>10}: {entity['word']}")
