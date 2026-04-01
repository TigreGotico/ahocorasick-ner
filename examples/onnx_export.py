"""Export Aho-Corasick automaton to ONNX for portable inference."""
from ahocorasick_ner.onnx_backend import OnnxAhocorasickNER

ner = OnnxAhocorasickNER()

# Build vocabulary
for city in ["New York", "London", "Tokyo", "Paris", "Berlin"]:
    ner.add_word("city", city)
for country in ["United States", "United Kingdom", "Japan", "France", "Germany"]:
    ner.add_word("country", country)

ner.fit()

# Save ONNX model + side tables
ner.save("/tmp/locations_onnx")
print("Exported: /tmp/locations_onnx.onnx + /tmp/locations_onnx.npz")

# Load and run inference (uses onnxruntime, zero Python in FSM traversal)
loaded = OnnxAhocorasickNER()
loaded.load("/tmp/locations_onnx")

text = "The summit in Paris brought leaders from Japan and Germany."
print(f"\n> {text}")
for entity in loaded.tag(text):
    print(f"  {entity['label']:>10}: {entity['word']}")
