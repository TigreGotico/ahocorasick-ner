import ahocorasick
import os
from ahocorasick_ner import AhocorasickNER

try:
    from datasets import load_dataset
except ImportError as e:
    # only used in demo classes, not a hard requirement
    def load_dataset(*args, **kwargs):
        raise e


class EncyclopediaMetallvmNER(AhocorasickNER):
    def __init__(self, path: str | None = None, case_sensitive: bool = False):
        super().__init__(case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.train(path)

    def train(self, path: str | None = None):
        self.load_huggingface()
        if path:
            self.save(path)

    def load_huggingface(self):
        dataset_name = "Jarbas/metal-archives-tracks"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            self.add_word("artist_name", entry["band_name"])
            if entry.get("track_name"):
                self.add_word("track_name", entry["track_name"])
            if entry.get("album_name"):
                self.add_word("album_name", entry["album_name"])
            self.add_word("album_type", entry["album_type"])

        dataset_name = "Jarbas/metal-archives-bands"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            self.add_word("artist_name", entry["name"])
            if entry.get("genre"):
                self.add_word("music_genre", entry["genre"])
            if entry.get("label"):
                self.add_word("record_label", entry["label"])


class MusicNER(AhocorasickNER):
    def __init__(self, path: str | None = None,
                 case_sensitive: bool = False):
        super().__init__(case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.train(path)

    def train(self, path: str | None = None):
        self.load_huggingface()
        if path:
            self.save(path)

    def load_huggingface(self):
        self.load_metallvm()
        self.load_jazz()
        self.load_prog()
        self.load_classical()
        self.load_trance()

    def load_trance(self):
        dataset_name = "Jarbas/trance_tracks"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("ARTIST(S)"):
                self.add_word("artist_name", entry["ARTIST(S)"])
            if entry.get("TRACK"):
                self.add_word("track_name", entry["TRACK"])
            if entry.get("STYLE"):
                self.add_word("music_genre", entry["STYLE"])

    def load_classical(self):
        dataset_name = "Jarbas/classic-composers"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("name"):
                self.add_word("artist_name", entry["name"])

    def load_prog(self):
        dataset_name = "Jarbas/prog-archives"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("artist"):
                self.add_word("artist_name", entry["artist"])
            if entry.get("genre"):
                self.add_word("music_genre", entry["genre"])

    def load_jazz(self):
        dataset_name = "Jarbas/jazz-music-archives"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("artist"):
                self.add_word("artist_name", entry["artist"])
            if entry.get("genre"):
                self.add_word("music_genre", entry["genre"])

    def load_metallvm(self):
        dataset_name = "Jarbas/metal-archives-tracks"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            self.add_word("artist_name", entry["band_name"])
            if entry.get("track_name"):
                self.add_word("track_name", entry["track_name"])
            if entry.get("album_name"):
                self.add_word("album_name", entry["album_name"])
            self.add_word("album_type", entry["album_type"])

        dataset_name = "Jarbas/metal-archives-bands"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            self.add_word("artist_name", entry["name"])
            if entry.get("genre"):
                self.add_word("music_genre", entry["genre"])
            if entry.get("label"):
                self.add_word("record_label", entry["label"])


class ImdbNER(AhocorasickNER):
    def __init__(self, path: str | None = None, case_sensitive: bool = False):
        super().__init__(case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.train(path)

    def train(self, path: str | None = None):
        self.load_huggingface()
        if path:
            self.save(path)

    def load_huggingface(self):
        dataset_name = "Jarbas/movie_actors"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("name"):
                self.add_word("movie_actor", entry["name"])

        dataset_name = "Jarbas/movie_directors"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("name"):
                self.add_word("movie_director", entry["name"])

        dataset_name = "Jarbas/movie_producers"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("name"):
                self.add_word("movie_producer", entry["name"])

        dataset_name = "Jarbas/movie_writers"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("name"):
                self.add_word("movie_writer", entry["name"])

        dataset_name = "Jarbas/movie_composers"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("name"):
                self.add_word("movie_composer", entry["name"])


if __name__ == "__main__":
    import time

    # e = MusicNER("media_net.ahocorasick")
    # e = ImdbNER("imdb.ahocorasick")
    e = EncyclopediaMetallvmNER("metallvm.ahocorasick")

    s = time.monotonic()
    for entity in e.tag("I fucking love black metal"):
        print(entity)
    print(time.monotonic() - s)
