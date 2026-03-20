# OpenVoiceOS Integration

Use ahocorasick-ner as a plugin in OpenVoiceOS skills.

---

## Overview

The library includes an **IntentTransformer** plugin for OpenVoiceOS that automatically extracts entities and injects them into the intent match context.

**Plugin:** `AhocorasickNERTransformer` — `ahocorasick_ner/opm.py:16`

---

## How It Works

### 1. Skill Registers Entities

```python
class MySkill(OVOSSkill):
    def initialize(self):
        # Register entities for recognition
        self.register_entity("artist_name", ["Metallica", "Iron Maiden"])
        self.register_entity("album", ["Master of Puppets"])
```

### 2. Plugin Listens

The transformer listens for `padatious:register_entity` messages on the message bus.

### 3. Plugin Extracts

When an utterance arrives, the plugin:
- Checks for registered entities
- Performs NER tagging
- Injects matches into intent context

### 4. Skill Accesses

Your intent handler receives matched entities:

```python
def handle_music_intent(self, message):
    entities = message.data.get("entities", [])
    for entity in entities:
        print(f"Matched: {entity['word']} ({entity['label']})")
```

---

## Setup

### Installation

```bash
# In your OVOS environment
uv pip install ahocorasick-ner
```

The plugin is automatically discovered via entry point:
```toml
[project.entry-points."opm.transformer.intent"]
ovos-ahocorasick-ner-plugin = "ahocorasick_ner.opm:AhocorasickNERTransformer"
```

### Enable in Config

Add to `~/.config/mycroft/mycroft.conf` or skill config:

```json
{
  "transformers": {
    "intent": {
      "ovos-ahocorasick-ner-plugin": {}
    }
  }
}
```

---

## Skill Integration Example

### Full Example: Music Recommendation Skill

```python
from ovos_workshop.skills.ovos import OVOSSkill
from ovos_workshop.decorators import intent_handler
from ovos_bus_client.message import Message

class MusicSkill(OVOSSkill):
    """Recommend music based on user preferences"""

    def initialize(self):
        """Called when skill loads"""
        # Register artist vocabulary
        self.register_entity("artist_name", [
            "Metallica",
            "Iron Maiden",
            "Black Sabbath",
            "AC/DC"
        ])

        # Register album vocabulary
        self.register_entity("album", [
            "Master of Puppets",
            "The Number of the Beast",
            "Paranoid",
            "Back in Black"
        ])

        self.log.info("Music entities registered")

    @intent_handler("play.music.intent")
    def handle_play_music(self, message):
        """User wants to play music"""
        # Extracted entities from message
        entities = message.data.get("entities", [])

        artist = None
        album = None

        # Parse extracted entities
        for entity in entities:
            if entity["label"] == "artist_name":
                artist = entity["word"]
            elif entity["label"] == "album":
                album = entity["word"]

        # Respond
        if artist and album:
            self.speak(f"Playing {album} by {artist}")
        elif artist:
            self.speak(f"Playing {artist}")
        elif album:
            self.speak(f"Playing {album}")
        else:
            self.speak("I didn't understand which music to play")

    def stop(self):
        pass

def create_skill():
    return MusicSkill()
```

### Corresponding Intent File

File: `vocab/en-us/play.music.intent`

```
play [some] music [by] {artist_name}
play {album}
play music from {artist_name}
i want to hear {artist_name}
put on {album}
```

---

## Entity Registration API

### register_entity

Register entities from a skill:

```python
# From a list
self.register_entity("artist_name", ["Metallica", "Iron Maiden"])

# From a file
self.register_entity("artist_name", file_name="vocab/en-us/artists.txt")

# With blacklisted words (excluded from matching)
self.register_entity(
    "artist_name",
    ["Metallica", "Iron Maiden"],
    blacklisted_words=["metal", "music"]
)
```

### register_vocab (alternative)

For non-entity intent matching:

```python
self.register_vocab("Metallica")
```

---

## Message Format

### Incoming Message

```python
{
    "type": "recognizer_loop:utterance",
    "data": {
        "utterances": ["play Metallica Master of Puppets"],
    },
    "context": {
        "skill_id": "my-music-skill"
    }
}
```

### After NER Processing

```python
{
    "type": "recognizer_loop:utterance",
    "data": {
        "utterances": ["play Metallica Master of Puppets"],
        "entities": [
            {
                "start": 5,
                "end": 14,
                "word": "Metallica",
                "label": "artist_name"
            },
            {
                "start": 16,
                "end": 33,
                "word": "Master of Puppets",
                "label": "album"
            }
        ]
    },
    "context": {
        "skill_id": "my-music-skill"
    }
}
```

---

## Configuration

### Per-Skill Config

Add to your skill's `settingsmeta.json` or config section:

```json
{
  "name": "Music Skill",
  "description": "Play music",
  "skillMetadata": {
    "sections": [
      {
        "name": "Entity Recognition",
        "fields": [
          {
            "name": "enable_ner",
            "type": "checkbox",
            "label": "Enable entity recognition",
            "value": true
          },
          {
            "name": "min_word_length",
            "type": "number",
            "label": "Minimum word length (chars)",
            "value": 5
          }
        ]
      }
    ]
  }
}
```

### Global Config

Add to `mycroft.conf` to configure all NER-enabled skills:

```json
{
  "transformers": {
    "intent": {
      "ovos-ahocorasick-ner-plugin": {
        "enabled": true,
        "min_word_len": 5,
        "case_sensitive": false
      }
    }
  }
}
```

---

## Debugging

### Check if Plugin Loaded

```python
from ovos_plugin_manager.templates.transformers import IntentTransformer

transformers = IntentTransformer.get_available_plugins()
print(transformers)  # Should include "ovos-ahocorasick-ner-plugin"
```

### Log Entity Matches

In your skill:

```python
def handle_play_music(self, message):
    entities = message.data.get("entities", [])
    self.log.debug(f"Extracted {len(entities)} entities: {entities}")

    for entity in entities:
        self.log.info(f"Entity: {entity['word']} ({entity['label']})")
```

### Check Bus Messages

Monitor the OVOS bus:

```bash
ovos-bus-client
# In another terminal:
# Your skill will emit messages, watch for "entities" in data
```

---

## Best Practices

### 1. Register Core Entities Only

Register entities that are core to your skill's intent matching:

```python
# ✅ DO: Register entities relevant to skill
self.register_entity("artist", ["Metallica", "Iron Maiden"])

# ❌ DON'T: Register every possible word
self.register_entity("word", ["the", "a", "is", ...])
```

### 2. Use Meaningful Labels

Choose descriptive entity labels:

```python
# ✅ Good: Specific labels
self.register_entity("artist_name", [...])
self.register_entity("song_title", [...])

# ❌ Bad: Generic labels
self.register_entity("entity1", [...])
self.register_entity("entity2", [...])
```

### 3. Keep Vocabularies Up-to-Date

Update entity lists as new items are added:

```python
def on_skill_update(self):
    """Called when skill updates"""
    # Re-register with new entities
    self.register_entity("artist", self.get_artists())
```

### 4. Fallback for Missing Entities

Gracefully handle cases where entities aren't extracted:

```python
def handle_music_intent(self, message):
    artist = None
    for entity in message.data.get("entities", []):
        if entity["label"] == "artist":
            artist = entity["word"]
            break

    if artist:
        self.play_artist(artist)
    else:
        # Fallback: ask user
        artist = self.get_response("Which artist?")
        if artist:
            self.play_artist(artist)
```

### 5. Test Entity Registration

Write tests to verify entity extraction:

```python
def test_artist_entity_extraction(self):
    """Test that artist entities are recognized"""
    msg = Message(
        "recognizer_loop:utterance",
        data={"utterances": ["play Metallica"]},
        context={"skill_id": self.skill.skill_id}
    )

    # Simulate transformer processing
    # Check that entities are populated
    assert msg.data.get("entities"), "No entities extracted"
```

---

## Troubleshooting

### Entities Not Extracted

1. **Check if plugin is loaded:**
   ```bash
   ovos-config show | grep ahocorasick
   ```

2. **Verify entities are registered:**
   ```python
   self.log.info(f"Registered entities: {self.entities}")
   ```

3. **Check utterance matches exactly:**
   - Whitespace and capitalization matter
   - "Metallica" != "metallica" (if case-sensitive)

### Performance Issues

If NER is slow:

1. **Reduce vocabulary size** — remove unused entities
2. **Increase min_word_len** — filter short matches
3. **Switch backends** — try NumPy or ONNX

```python
# In transformer config
{
  "ovos-ahocorasick-ner-plugin": {
    "min_word_len": 6,  # Increase from 5
    "backend": "numpy"  # Use NumPy instead of pyahocorasick
  }
}
```

---

## See Also

- **[API Reference](api-reference.md)** — Full method documentation
- **[Examples](examples.md)** — Usage patterns
- **[OVOS Documentation](https://openvoiceos.github.io/)** — Main OVOS docs
