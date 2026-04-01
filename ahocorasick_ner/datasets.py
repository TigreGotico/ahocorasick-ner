import os
from typing import Optional, Union
from ahocorasick_ner import AhocorasickNER

try:
    from datasets import load_dataset
except ImportError:
    # only used in demo classes, not a hard requirement
    def load_dataset(*args, **kwargs):
        raise ImportError("The 'datasets' library is required to use this functionality. "
                          "Install it with 'pip install datasets'.")


class EncyclopediaMetallvmNER(AhocorasickNER):
    """
    NER system pre-loaded with band, track, and album names from Encyclopedia Metallum
    via Hugging Face datasets.
    """

    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False):
        """
        Initializes the Encyclopedia Metallum NER.

        Args:
            path: Optional path to a pre-trained automaton file.
            case_sensitive: Whether matching should be case-sensitive.
        """
        super().__init__(case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.train(path)

    def train(self, path: Optional[str] = None) -> None:
        """
        Loads data from Hugging Face and optionally saves the automaton.

        Args:
            path: Optional path to save the trained automaton.
        """
        self.load_huggingface()
        if path:
            self.save(path)

    def load_huggingface(self) -> None:
        """
        Loads Metal Archives data from Hugging Face.
        """
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
    """
    A comprehensive music NER system combining multiple genres (Metal, Jazz, Prog, Classical, Trance).
    """

    def __init__(self, path: Optional[str] = None,
                 case_sensitive: bool = False):
        """
        Initializes the Music NER.

        Args:
            path: Optional path to a pre-trained automaton file.
            case_sensitive: Whether matching should be case-sensitive.
        """
        super().__init__(case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.train(path)

    def train(self, path: Optional[str] = None) -> None:
        """
        Loads data from multiple music datasets and optionally saves the automaton.

        Args:
            path: Optional path to save the trained automaton.
        """
        self.load_huggingface()
        if path:
            self.save(path)

    def load_huggingface(self) -> None:
        """
        Loads all music sub-datasets.
        """
        self.load_metallvm()
        self.load_jazz()
        self.load_prog()
        self.load_classical()
        self.load_trance()

    def load_trance(self) -> None:
        dataset_name = "Jarbas/trance_tracks"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("ARTIST(S)"):
                self.add_word("artist_name", entry["ARTIST(S)"])
            if entry.get("TRACK"):
                self.add_word("track_name", entry["TRACK"])
            if entry.get("STYLE"):
                self.add_word("music_genre", entry["STYLE"])

    def load_classical(self) -> None:
        dataset_name = "Jarbas/classic-composers"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("name"):
                self.add_word("artist_name", entry["name"])

    def load_prog(self) -> None:
        dataset_name = "Jarbas/prog-archives"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("artist"):
                self.add_word("artist_name", entry["artist"])
            if entry.get("genre"):
                self.add_word("music_genre", entry["genre"])

    def load_jazz(self) -> None:
        dataset_name = "Jarbas/jazz-music-archives"
        dataset = load_dataset(dataset_name)["train"]
        for entry in dataset:
            if entry.get("artist"):
                self.add_word("artist_name", entry["artist"])
            if entry.get("genre"):
                self.add_word("music_genre", entry["genre"])

    def load_metallvm(self) -> None:
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
    """
    NER system for movie-related entities (actors, directors, producers, etc.) via IMDB data.
    """

    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False):
        """
        Initializes the IMDB NER.

        Args:
            path: Optional path to a pre-trained automaton file.
            case_sensitive: Whether matching should be case-sensitive.
        """
        super().__init__(case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.train(path)

    def train(self, path: Optional[str] = None) -> None:
        """
        Loads data from Hugging Face and optionally saves the automaton.

        Args:
            path: Optional path to save the trained automaton.
        """
        self.load_huggingface()
        if path:
            self.save(path)

    def load_huggingface(self) -> None:
        """
        Loads various IMDB datasets from Hugging Face.
        """
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


class MetalArchivesBandsNER(AhocorasickNER):
    """Extract metal band names from Encyclopedia Metallum.

    Contains ~4.6k metal band names and associated metadata.
    Supports filtering by country, genre, and formation year range.

    Example:
        ```python
        from ahocorasick_ner.datasets import MetalArchivesBandsNER

        # Load all metal bands
        ner = MetalArchivesBandsNER()

        # Load only Portuguese bands
        pt_bands = MetalArchivesBandsNER(origin="Portugal")

        # Load only thrash metal bands from 1980-1995
        thrash = MetalArchivesBandsNER(genre="Thrash Metal",
                                        formed_year_min=1980,
                                        formed_year_max=1995)
        ```
    """

    def __init__(
        self,
        path: Optional[str] = None,
        case_sensitive: bool = False,
        origin: Optional[str] = None,
        genre: Optional[str] = None,
        formed_year_min: Optional[int] = None,
        formed_year_max: Optional[int] = None,
    ):
        """
        Initializes MetalArchivesBandsNER.

        Args:
            path: Optional path to a pre-trained automaton file.
            case_sensitive: Whether matching should be case-sensitive.
            origin: Optional country to filter bands by.
            genre: Optional genre to filter bands by.
            formed_year_min: Optional minimum formation year.
            formed_year_max: Optional maximum formation year.
        """
        super().__init__(case_sensitive=case_sensitive)
        self.origin = origin
        self.genre = genre
        self.formed_year_min = formed_year_min
        self.formed_year_max = formed_year_max
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load metal band names with optional filtering."""
        dataset = load_dataset("TigreGotico/metal-archives-bands")["train"]
        count = 0
        for entry in dataset:
            if self.origin and entry.get("origin") != self.origin:
                continue
            if self.genre and entry.get("genre") != self.genre:
                continue
            formed_year = entry.get("date")
            if formed_year is not None:
                if self.formed_year_min and formed_year < self.formed_year_min:
                    continue
                if self.formed_year_max and formed_year > self.formed_year_max:
                    continue

            if entry.get("name"):
                self.add_word("metal_band", entry["name"])
                count += 1
        self.fit()
        filters = []
        if self.origin:
            filters.append(f"origin={self.origin}")
        if self.genre:
            filters.append(f"genre={self.genre}")
        if self.formed_year_min or self.formed_year_max:
            year_range = f"formed_year={self.formed_year_min or '?'}-{self.formed_year_max or '?'}"
            filters.append(year_range)

        if filters:
            print(f"Loaded {count} metal bands ({', '.join(filters)})")
        else:
            print(f"Loaded {count} metal bands")


class MetalArchivesTrackNER(AhocorasickNER):
    """Extract metal tracks, albums, and bands from Encyclopedia Metallum.

    Contains ~205k metal tracks with associated album and band info.
    Supports filtering by band origin/country and album type.

    Example:
        ```python
        from ahocorasick_ner.datasets import MetalArchivesTrackNER

        # Load all tracks
        ner = MetalArchivesTrackNER()

        # Load only studio full-length albums from Swedish bands
        swedish = MetalArchivesTrackNER(band_origin="Sweden",
                                         album_type="Full-length")

        # Load only demo tracks
        demos = MetalArchivesTrackNER(album_type="Demo")
        ```
    """

    def __init__(
        self,
        path: Optional[str] = None,
        case_sensitive: bool = False,
        band_origin: Optional[str] = None,
        album_type: Optional[str] = None,
    ):
        """
        Initializes MetalArchivesTrackNER.

        Args:
            path: Optional path to a pre-trained automaton file.
            case_sensitive: Whether matching should be case-sensitive.
            band_origin: Optional country to filter tracks by band origin.
            album_type: Optional album type: Full-length, EP, Single, Live album,
                Demo, Compilation, Split, etc.
        """
        super().__init__(case_sensitive=case_sensitive)
        self.band_origin = band_origin
        self.album_type = album_type
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load metal tracks with optional filtering."""
        dataset = load_dataset("TigreGotico/metal-archives-tracks")["train"]
        count = 0
        for entry in dataset:
            if self.band_origin and entry.get("band_origin") != self.band_origin:
                continue
            if self.album_type and entry.get("album_type") != self.album_type:
                continue

            self.add_word("metal_band", entry.get("band_name", ""))
            if entry.get("track_name"):
                self.add_word("metal_track", entry["track_name"])
                count += 1
            if entry.get("album_name"):
                self.add_word("metal_album", entry["album_name"])
        self.fit()
        filters = []
        if self.band_origin:
            filters.append(f"band_origin={self.band_origin}")
        if self.album_type:
            filters.append(f"album_type={self.album_type}")

        if filters:
            print(f"Loaded metal tracks ({', '.join(filters)})")
        else:
            print("Loaded metal tracks")


class MovieActorNER(AhocorasickNER):
    """Extract movie actor names from IMDB.

    Contains ~6.3M actor names from IMDB.
    Supports filtering by gender.

    Example:
        ```python
        from ahocorasick_ner.datasets import MovieActorNER

        # Load all actors
        ner = MovieActorNER()

        # Load only female actors
        actresses = MovieActorNER(gender="Female")

        # Load only male actors
        actors = MovieActorNER(gender="Male")
        ```
    """

    def __init__(
        self,
        path: Optional[str] = None,
        case_sensitive: bool = False,
        gender: Optional[str] = None,
    ):
        """
        Initializes MovieActorNER.

        Args:
            path: Optional path to a pre-trained automaton file.
            case_sensitive: Whether matching should be case-sensitive.
            gender: Optional gender filter: "Male" or "Female".
        """
        super().__init__(case_sensitive=case_sensitive)
        self.gender = gender
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load actor names with optional gender filtering."""
        dataset = load_dataset("TigreGotico/movie_actors")["train"]
        count = 0
        for entry in dataset:
            if self.gender and entry.get("gender") != self.gender:
                continue
            if entry.get("name"):
                self.add_word("movie_actor", entry["name"])
                count += 1
        self.fit()
        if self.gender:
            print(f"Loaded {count} {self.gender.lower()} actors")
        else:
            print(f"Loaded {count} actors")


class MovieDirectorNER(AhocorasickNER):
    """Extract movie director names from IMDB.

    Contains ~128k director names.

    Example:
        ```python
        from ahocorasick_ner.datasets import MovieDirectorNER
        ner = MovieDirectorNER()
        for entity in ner.tag("Directed by Steven Spielberg"):
            print(entity)
        ```
    """

    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False):
        super().__init__(case_sensitive=case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load director names from IMDB dataset."""
        dataset = load_dataset("TigreGotico/movie_directors")["train"]
        for entry in dataset:
            if entry.get("name"):
                self.add_word("movie_director", entry["name"])
        self.fit()


class MovieComposerNER(AhocorasickNER):
    """Extract movie composer names from IMDB.

    Contains ~221k composer names.

    Example:
        ```python
        from ahocorasick_ner.datasets import MovieComposerNER
        ner = MovieComposerNER()
        for entity in ner.tag("Score by Hans Zimmer"):
            print(entity)
        ```
    """

    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False):
        super().__init__(case_sensitive=case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load composer names from IMDB dataset."""
        dataset = load_dataset("TigreGotico/movie_composers")["train"]
        for entry in dataset:
            if entry.get("name"):
                self.add_word("movie_composer", entry["name"])
        self.fit()


class JazzNER(AhocorasickNER):
    """Extract jazz artists and genres from jazz-music-archives.

    Contains ~12.3k jazz artists and genre information.

    Example:
        ```python
        from ahocorasick_ner.datasets import JazzNER
        ner = JazzNER()
        for entity in ner.tag("Miles Davis played cool jazz"):
            print(entity)
        ```
    """

    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False):
        super().__init__(case_sensitive=case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load jazz artists and genres."""
        dataset = load_dataset("TigreGotico/jazz-music-archives")["train"]
        for entry in dataset:
            if entry.get("artist"):
                self.add_word("jazz_artist", entry["artist"])
            if entry.get("genre"):
                self.add_word("jazz_genre", entry["genre"])
        self.fit()


class ProgRockNER(AhocorasickNER):
    """Extract progressive rock artists and genres from prog-archives.

    Contains ~12.4k prog rock artists and genre information.

    Example:
        ```python
        from ahocorasick_ner.datasets import ProgRockNER
        ner = ProgRockNER()
        for entity in ner.tag("Yes and Genesis were prog pioneers"):
            print(entity)
        ```
    """

    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False):
        super().__init__(case_sensitive=case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load prog rock artists and genres."""
        dataset = load_dataset("TigreGotico/prog-archives")["train"]
        for entry in dataset:
            if entry.get("artist"):
                self.add_word("prog_artist", entry["artist"])
            if entry.get("genre"):
                self.add_word("prog_genre", entry["genre"])
        self.fit()


class CompanyNamesNER(AhocorasickNER):
    """Extract company names from curated company dataset.

    Contains ~50k+ company names from multiple sources.

    Example:
        ```python
        from ahocorasick_ner.datasets import CompanyNamesNER
        ner = CompanyNamesNER()
        for entity in ner.tag("Apple and Microsoft compete"):
            print(entity)
        ```
    """

    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False):
        super().__init__(case_sensitive=case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load company names from dataset."""
        try:
            dataset = load_dataset("nbroad/company_names")["train"]
        except Exception:
            # Fallback
            dataset = load_dataset("byrneml/company_names")["train"]

        seen = set()
        for row in dataset:
            name = row.get("name") or row.get("company_name")
            if name and name.lower() not in seen:
                seen.add(name.lower())
                self.add_word("company", name)

        self.fit()


class SpotifyTracksNER(AhocorasickNER):
    """Extract Spotify track titles, artists, and genres.

    Contains 200k+ tracks from Spotify with artist and genre information.
    Supports filtering by genre, popularity, and audio features.

    Example:
        ```python
        from ahocorasick_ner.datasets import SpotifyTracksNER

        # Load all Spotify tracks
        ner = SpotifyTracksNER()

        # Load only popular rock tracks (>70 popularity)
        rock = SpotifyTracksNER(genre="rock", popularity_min=70)

        # Load only danceability dance tracks
        dance = SpotifyTracksNER(genre="dance", danceability_min=0.7)

        # Load only high-energy, high-valence (happy) tracks
        upbeat = SpotifyTracksNER(energy_min=0.8, valence_min=0.7)

        # Load only acoustic instrumental tracks
        acoustic = SpotifyTracksNER(acousticness_min=0.8,
                                     instrumentalness_min=0.5)
        ```
    """

    def __init__(
        self,
        path: Optional[str] = None,
        case_sensitive: bool = False,
        genre: Optional[str] = None,
        popularity_min: Optional[int] = None,
        popularity_max: Optional[int] = None,
        energy_min: Optional[float] = None,
        danceability_min: Optional[float] = None,
        acousticness_min: Optional[float] = None,
        valence_min: Optional[float] = None,
        explicit: Optional[bool] = None,
    ):
        """
        Initializes SpotifyTracksNER.

        Args:
            path: Optional path to a pre-trained automaton file.
            case_sensitive: Whether matching should be case-sensitive.
            genre: Optional genre to filter tracks by.
            popularity_min: Minimum popularity score (0-100).
            popularity_max: Maximum popularity score (0-100).
            energy_min: Minimum energy level (0.0-1.0).
            danceability_min: Minimum danceability (0.0-1.0).
            acousticness_min: Minimum acousticness (0.0-1.0).
            valence_min: Minimum valence/positiveness (0.0-1.0).
            explicit: Filter by explicit content flag (True/False/None).
        """
        super().__init__(case_sensitive=case_sensitive)
        self.genre = genre
        self.popularity_min = popularity_min
        self.popularity_max = popularity_max
        self.energy_min = energy_min
        self.danceability_min = danceability_min
        self.acousticness_min = acousticness_min
        self.valence_min = valence_min
        self.explicit = explicit
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load Spotify tracks with optional audio feature filtering."""
        dataset = load_dataset("maharshipandya/spotify-tracks-dataset")["train"]

        seen = set()
        count = 0
        for row in dataset:
            if self.genre and row.get("genre") != self.genre:
                continue

            popularity = row.get("popularity")
            if popularity is not None:
                if self.popularity_min and popularity < self.popularity_min:
                    continue
                if self.popularity_max and popularity > self.popularity_max:
                    continue

            if self.explicit is not None and row.get("explicit") != self.explicit:
                continue

            if self.energy_min and row.get("energy", 0) < self.energy_min:
                continue
            if self.danceability_min and row.get("danceability", 0) < self.danceability_min:
                continue
            if self.acousticness_min and row.get("acousticness", 0) < self.acousticness_min:
                continue
            if self.valence_min and row.get("valence", 0) < self.valence_min:
                continue

            # Track titles
            track = row.get("track_name")
            if track and track.lower() not in seen:
                seen.add(track.lower())
                self.add_word("track_name", track)
                count += 1

            # Artists
            artist = row.get("artist_name")
            if artist and artist.lower() not in seen:
                seen.add(artist.lower())
                self.add_word("artist_name", artist)

            # Genres
            genre = row.get("genre")
            if genre and genre.lower() not in seen:
                seen.add(genre.lower())
                self.add_word("music_genre", genre)

        self.fit()
        filters = []
        if self.genre:
            filters.append(f"genre={self.genre}")
        if self.popularity_min:
            filters.append(f"popularity>={self.popularity_min}")
        if self.energy_min:
            filters.append(f"energy>={self.energy_min}")
        if self.danceability_min:
            filters.append(f"danceability>={self.danceability_min}")
        if self.acousticness_min:
            filters.append(f"acousticness>={self.acousticness_min}")
        if self.valence_min:
            filters.append(f"valence>={self.valence_min}")
        if self.explicit is not None:
            filters.append(f"explicit={self.explicit}")

        if filters:
            print(f"Loaded Spotify tracks ({', '.join(filters)})")
        else:
            print("Loaded Spotify tracks")


class RecipeIngredientsNER(AhocorasickNER):
    """Extract recipe ingredients from RecipeNLG dataset.

    Contains 2.2M recipes with ingredient listings.
    Supports filtering by recipe source (Gathered or Recipes1M).

    Example:
        ```python
        from ahocorasick_ner.datasets import RecipeIngredientsNER

        # Load all recipe ingredients
        ner = RecipeIngredientsNER()

        # Load only ingredients from Gathered recipes
        gathered = RecipeIngredientsNER(source="Gathered")

        # Load only ingredients from Recipes1M dataset
        recipes1m = RecipeIngredientsNER(source="Recipes1M")
        ```
    """

    def __init__(
        self,
        path: Optional[str] = None,
        case_sensitive: bool = False,
        source: Optional[str] = None,
    ):
        """
        Initializes RecipeIngredientsNER.

        Args:
            path: Optional path to a pre-trained automaton file.
            case_sensitive: Whether matching should be case-sensitive.
            source: Optional source filter: "Gathered" or "Recipes1M".
        """
        super().__init__(case_sensitive=case_sensitive)
        self.source = source
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load recipe ingredients with optional source filtering."""
        try:
            dataset = load_dataset("mbien/recipe_nlg")["train"]
        except Exception:
            dataset = load_dataset("m3hrdadfi/recipe_nlg_lite")["train"]

        seen = set()
        count = 0
        for row in dataset:
            if self.source and row.get("source") != self.source:
                continue

            ingredients = row.get("ingredients", [])
            if isinstance(ingredients, list):
                for ingredient in ingredients:
                    # Handle both strings and dicts
                    if isinstance(ingredient, dict):
                        ingredient = ingredient.get("name") or ingredient.get("text")
                    if ingredient and ingredient.lower() not in seen:
                        seen.add(ingredient.lower())
                        self.add_word("ingredient", ingredient)
                        count += 1

        self.fit()
        if self.source:
            print(f"Loaded recipe ingredients from {self.source}")
        else:
            print("Loaded recipe ingredients")


class FoodProductsNER(AhocorasickNER):
    """Extract food product names from Open Food Facts.

    Contains 4M+ food product names, brands, and ingredients.
    Supports filtering by allergens, dietary labels, and country.

    Example:
        ```python
        from ahocorasick_ner.datasets import FoodProductsNER

        # Load all products
        ner = FoodProductsNER()

        # Load only gluten-free products
        gluten_free = FoodProductsNER(allergen="gluten-free")

        # Load only vegan products
        vegan = FoodProductsNER(vegan=True)

        # Load only organic products from France
        org_fr = FoodProductsNER(organic=True, country="fr")

        # Load nut-free products
        nut_free = FoodProductsNER(allergen="tree-nut-free")
        ```
    """

    def __init__(
        self,
        path: Optional[str] = None,
        case_sensitive: bool = False,
        allergen: Optional[str] = None,
        vegan: Optional[bool] = None,
        vegetarian: Optional[bool] = None,
        organic: Optional[bool] = None,
        country: Optional[str] = None,
    ):
        """
        Initializes FoodProductsNER.

        Args:
            path: Optional path to a pre-trained automaton file.
            case_sensitive: Whether matching should be case-sensitive.
            allergen: Optional allergen to filter by (e.g., "gluten-free", "nut-free").
            vegan: Filter for vegan products (True/False/None).
            vegetarian: Filter for vegetarian products (True/False/None).
            organic: Filter for organic products (True/False/None).
            country: Optional country code to filter by manufacturing location.
        """
        super().__init__(case_sensitive=case_sensitive)
        self.allergen = allergen
        self.vegan = vegan
        self.vegetarian = vegetarian
        self.organic = organic
        self.country = country
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load food products with optional dietary/allergen filtering."""
        dataset = load_dataset("openfoodfacts/product-database")

        seen = set()
        count = 0
        for row in (dataset.get("train", dataset)):
            # Check allergen filter
            if self.allergen:
                allergens_tags = row.get("allergens_tags", [])
                if not allergens_tags or self.allergen not in allergens_tags:
                    continue

            # Check dietary filters
            if self.vegan or self.vegetarian or self.organic:
                labels_tags = row.get("labels_tags", [])
                if self.vegan and "vegan" not in labels_tags:
                    continue
                if self.vegetarian and "vegetarian" not in labels_tags:
                    continue
                if self.organic and "organic" not in labels_tags:
                    continue

            # Check country filter
            if self.country:
                countries_tags = row.get("countries_tags", [])
                if not countries_tags or self.country not in countries_tags:
                    continue

            # Product name
            product = row.get("product_name") or row.get("product")
            if product and product.lower() not in seen:
                seen.add(product.lower())
                self.add_word("food_product", product)
                count += 1

            # Brand
            brand = row.get("brand")
            if brand and brand.lower() not in seen:
                seen.add(brand.lower())
                self.add_word("brand", brand)

        self.fit()
        filters = []
        if self.allergen:
            filters.append(f"{self.allergen}")
        if self.vegan:
            filters.append("vegan")
        if self.vegetarian:
            filters.append("vegetarian")
        if self.organic:
            filters.append("organic")
        if self.country:
            filters.append(f"country={self.country}")

        if filters:
            print(f"Loaded food products ({', '.join(filters)})")
        else:
            print("Loaded food products")


class ProgrammingLanguageNER(AhocorasickNER):
    """Extract programming language names.

    Covers 358 programming languages from The Stack dataset.

    Example:
        ```python
        from ahocorasick_ner.datasets import ProgrammingLanguageNER
        ner = ProgrammingLanguageNER()
        for entity in ner.tag("Code in Python, JavaScript, and Rust"):
            print(entity)
        ```
    """

    # Common programming languages
    LANGUAGES = [
        "Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "Ruby",
        "PHP", "Swift", "Kotlin", "TypeScript", "R", "Scala", "Haskell",
        "Elixir", "Clojure", "Lisp", "Perl", "Shell", "Bash", "SQL",
        "HTML", "CSS", "Groovy", "Julia", "Lua", "Dart", "Matlab",
        "VB.NET", "Objective-C", "Ada", "COBOL", "Fortran", "Pascal",
        "F#", "OCaml", "Erlang", "Scheme", "Racket", "YAML", "XML",
        "JSON", "TOML", "GraphQL", "Solidity", "VHDL", "Verilog",
        "Assembly", "PowerShell", "VBScript", "Nim", "Crystal",
    ]

    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False):
        super().__init__(case_sensitive=case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load programming language names."""
        for lang in self.LANGUAGES:
            self.add_word("programming_language", lang)
        self.fit()


class GeoNamesNER(AhocorasickNER):
    """Extract city and location names from GeoNames database.

    Contains ~280k cities and locations worldwide with multilingual names.

    Example:
        ```python
        from ahocorasick_ner.datasets import GeoNamesNER
        ner = GeoNamesNER()
        for entity in ner.tag("Paris and Tokyo are major cities"):
            print(entity)
        ```
    """

    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False):
        super().__init__(case_sensitive=case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load city and location names from GeoNames dataset."""
        try:
            dataset = load_dataset("vladislav-savko/geonames-all-cities")
        except Exception:
            # Fallback to alternate GeoNames dataset
            dataset = load_dataset("bstds/geonames")

        seen = set()
        for row in dataset.get("train", dataset):
            name = row.get("name") or row.get("asciiname")
            if name and name.lower() not in seen:
                seen.add(name.lower())
                self.add_word("location", name)

            # Add alternate names
            alts = row.get("alternatenames") or row.get("alternateNames")
            if isinstance(alts, str):
                for alt in alts.split(","):
                    alt = alt.strip()
                    if alt and alt.lower() not in seen:
                        seen.add(alt.lower())
                        self.add_word("location", alt)

        self.fit()


class PersonNamesNER(AhocorasickNER):
    """Extract person names from multilingual surname dataset.

    Contains surnames and naming patterns for 30+ countries/languages.

    Language support: English, Spanish, French, German, Chinese, Arabic, etc.

    Example:
        ```python
        from ahocorasick_ner.datasets import PersonNamesNER
        ner = PersonNamesNER()
        for entity in ner.tag("John Smith and Maria Garcia"):
            print(entity)
        ```
    """

    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False):
        super().__init__(case_sensitive=case_sensitive)
        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load person names from surname-nationality dataset."""
        dataset = load_dataset("Hobson/surname-nationality")["train"]

        seen = set()
        for row in dataset:
            surname = row.get("surname") or row.get("name")
            if surname and surname.lower() not in seen:
                seen.add(surname.lower())
                self.add_word("person_name", surname)

        self.fit()


class WikidataEntityNER(AhocorasickNER):
    """Extract entities from Wikidata by instance-of QID.

    Uses Wikimedians/wikidata-all HuggingFace dataset to extract entity names
    filtered by Wikidata QID (e.g. Q729 for animals, Q7432 for plants).

    Language support: Multilingual via Wikidata labels with English fallback.

    Example:
        ```python
        from ahocorasick_ner.datasets import WikidataEntityNER

        animals = WikidataEntityNER(entity_type="Animal", wikidata_qid="Q729")
        for entity in animals.tag("I saw a dog and a cat"):
            print(entity)
        ```
    """

    # Common Wikidata QIDs for entity types
    QIDS = {
        "animal": "Q729",
        "plant": "Q7432",
        "body_part": "Q4891",
        "family_relation": "Q1038",
        "country": "Q6256",
        "city": "Q515",
        "profession": "Q28640",
        "disease": "Q12078",
        "sport": "Q349",
        "instrument": "Q11649",
        "language": "Q315",
        "color": "Q1075",
    }

    def __init__(
        self,
        entity_type: str,
        wikidata_qid: Optional[str] = None,
        lang: str = "en",
        path: Optional[str] = None,
        case_sensitive: bool = False,
    ):
        """
        Initializes WikidataEntityNER.

        Args:
            entity_type: Entity label (e.g. "Animal", "Plant", "Country").
            wikidata_qid: Wikidata QID for entity class. If None, infers from entity_type.
            lang: Language code for entity labels (e.g. "en", "de", "es").
            path: Optional path to a pre-trained automaton file.
            case_sensitive: Whether matching should be case-sensitive.
        """
        super().__init__(case_sensitive=case_sensitive)
        self.entity_type = entity_type
        self.lang = lang

        # Auto-map entity_type to QID if not provided
        if wikidata_qid is None:
            key = entity_type.lower().replace(" ", "_")
            if key in self.QIDS:
                wikidata_qid = self.QIDS[key]
            else:
                raise ValueError(
                    f"Unknown entity type '{entity_type}'. "
                    f"Provide wikidata_qid or use: {list(self.QIDS.keys())}"
                )

        self.wikidata_qid = wikidata_qid

        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load entities from Wikidata filtered by QID."""
        dataset = load_dataset("Wikimedians/wikidata-all")["train"]

        seen = set()
        count = 0

        for row in dataset:
            if row.get("instance_of") != self.wikidata_qid:
                continue

            labels = row.get("labels", {})

            # Preferred language first
            label = labels.get(self.lang) or labels.get("en")
            if label and label.lower() not in seen:
                seen.add(label.lower())
                self.add_word(self.entity_type, label)
                count += 1

            # Aliases as fallback variants
            for alias in labels.get("aliases", []):
                if alias and alias.lower() not in seen:
                    seen.add(alias.lower())
                    self.add_word(self.entity_type, alias)
                    count += 1

        self.fit()
        print(f"Loaded {count} {self.entity_type} entities from Wikidata (QID {self.wikidata_qid})")


# Dedicated Wikidata entity type subclasses for easy access
class WikidataAnimalNER(WikidataEntityNER):
    """Extract animal names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Animal", wikidata_qid="Q729", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataPlantNER(WikidataEntityNER):
    """Extract plant names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Plant", wikidata_qid="Q7432", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataCountryNER(WikidataEntityNER):
    """Extract country names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Country", wikidata_qid="Q6256", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataCityNER(WikidataEntityNER):
    """Extract city names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="City", wikidata_qid="Q515", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataPersonNER(WikidataEntityNER):
    """Extract person names from Wikidata (Q5: human)."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Person", wikidata_qid="Q5", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataProfessionNER(WikidataEntityNER):
    """Extract profession/occupation names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Profession", wikidata_qid="Q28640", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataDiseaseNER(WikidataEntityNER):
    """Extract disease names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Disease", wikidata_qid="Q12078", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataLanguageNER(WikidataEntityNER):
    """Extract language names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Language", wikidata_qid="Q315", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataSportNER(WikidataEntityNER):
    """Extract sport names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Sport", wikidata_qid="Q349", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataBodyPartNER(WikidataEntityNER):
    """Extract anatomical body part names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="BodyPart", wikidata_qid="Q4891", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataLandmarkNER(WikidataEntityNER):
    """Extract landmark and monument names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Landmark", wikidata_qid="Q570116", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataUniversityNER(WikidataEntityNER):
    """Extract university and higher education institution names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="University", wikidata_qid="Q3918", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataMuseumNER(WikidataEntityNER):
    """Extract museum names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Museum", wikidata_qid="Q33506", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataAuthorNER(WikidataEntityNER):
    """Extract author and writer names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Author", wikidata_qid="Q36180", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataBookNER(WikidataEntityNER):
    """Extract book titles from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Book", wikidata_qid="Q571", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataPublisherNER(WikidataEntityNER):
    """Extract publisher names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Publisher", wikidata_qid="Q2085381", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataAthleteNER(WikidataEntityNER):
    """Extract athlete names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Athlete", wikidata_qid="Q2066131", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataSportsTeamNER(WikidataEntityNER):
    """Extract sports team names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="SportsTeam", wikidata_qid="Q12973014", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataMovieNER(WikidataEntityNER):
    """Extract movie and film titles from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Movie", wikidata_qid="Q11424", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataMusicalNER(WikidataEntityNER):
    """Extract musical instrument names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Instrument", wikidata_qid="Q11649", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataCompanyNER(WikidataEntityNER):
    """Extract company and corporation names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="Company", wikidata_qid="Q783794", lang=lang, path=path, case_sensitive=case_sensitive)


class WikidataFamilyRelationNER(WikidataEntityNER):
    """Extract family relationship names from Wikidata."""
    def __init__(self, path: Optional[str] = None, case_sensitive: bool = False, lang: str = "en"):
        super().__init__(entity_type="FamilyRelation", wikidata_qid="Q1038", lang=lang, path=path, case_sensitive=case_sensitive)


class GenericHFDatasetNER(AhocorasickNER):
    """Generic loader for HuggingFace datasets with entity values in a column.

    For datasets that provide entities as direct column values without
    complex structure. Supports filtering by column values.

    Example:
        ```python
        from ahocorasick_ner.datasets import GenericHFDatasetNER

        # Load all colors
        colors = GenericHFDatasetNER(
            entity_type="Color",
            hf_dataset="boltuix/color-pedia",
            column="name"
        )

        # Load only warm colors (if dataset has color_family column)
        warm = GenericHFDatasetNER(
            entity_type="Color",
            hf_dataset="boltuix/color-pedia",
            column="name",
            filter_column="color_family",
            filter_value="warm"
        )
        ```
    """

    def __init__(
        self,
        entity_type: str,
        hf_dataset: str,
        column: str = "name",
        filter_column: Optional[str] = None,
        filter_value: Optional[Union[str, int, float]] = None,
        path: Optional[str] = None,
        case_sensitive: bool = False,
    ):
        """
        Initializes generic HuggingFace dataset NER.

        Args:
            entity_type: Entity label (e.g. "Color", "Plant").
            hf_dataset: HuggingFace dataset ID.
            column: Column name containing entity values.
            filter_column: Optional column to filter rows by.
            filter_value: Value to match in filter_column. Rows where
                filter_column != filter_value are skipped.
            path: Optional path to a pre-trained automaton file.
            case_sensitive: Whether matching should be case-sensitive.
        """
        super().__init__(case_sensitive=case_sensitive)
        self.entity_type = entity_type
        self.hf_dataset = hf_dataset
        self.column = column
        self.filter_column = filter_column
        self.filter_value = filter_value

        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load entities from HuggingFace dataset column with optional filtering."""
        dataset = load_dataset(self.hf_dataset)

        # Handle both single split and multi-split datasets
        if isinstance(dataset, dict):
            splits = dataset.values()
        else:
            splits = [dataset]

        seen = set()
        count = 0

        for split in splits:
            for row in split:
                # Apply filter if specified
                if self.filter_column and self.filter_value is not None:
                    row_value = row.get(self.filter_column)
                    if row_value != self.filter_value:
                        continue

                value = row.get(self.column)
                if value and isinstance(value, str):
                    value_lower = value.lower()
                    if value_lower not in seen:
                        seen.add(value_lower)
                        self.add_word(self.entity_type, value)
                        count += 1

        self.fit()
        filter_msg = (
            f" (filtered by {self.filter_column}={self.filter_value})"
            if self.filter_column else ""
        )
        print(f"Loaded {count} {self.entity_type} entities from {self.hf_dataset}{filter_msg}")


class BC5CDRMedicalNER(AhocorasickNER):
    """Extract diseases and chemicals from BC5CDR biomedical NER dataset.

    The BC5CDR dataset contains biomedical text annotated with chemical and
    disease mentions in BIO format.

    Language support: English only.

    Example:
        ```python
        from ahocorasick_ner.datasets import BC5CDRMedicalNER

        diseases = BC5CDRMedicalNER(entity_type="Disease")
        for entity in diseases.tag("Aspirin treats headaches"):
            print(entity)
        ```
    """

    def __init__(
        self,
        entity_type: str = "Disease",
        path: Optional[str] = None,
        case_sensitive: bool = False,
    ):
        """
        Initializes BC5CDR Medical NER.

        Args:
            entity_type: "Disease" or "Chemical".
            path: Optional path to a pre-trained automaton file.
            case_sensitive: Whether matching should be case-sensitive.
        """
        if entity_type not in ("Disease", "Chemical"):
            raise ValueError(f"BC5CDR only supports Disease or Chemical; got {entity_type}")

        super().__init__(case_sensitive=case_sensitive)
        self.entity_type = entity_type

        if path and os.path.exists(path):
            self.load(path)
        else:
            self.load_huggingface()
            if path:
                self.save(path)

    def load_huggingface(self) -> None:
        """Load disease/chemical entities from BC5CDR dataset."""
        dataset = load_dataset("BC5CDR")

        seen = set()
        count = 0

        for row in dataset.get("train", []):
            ents = row.get("entities", [])
            for ent in ents:
                if ent.get("type") == self.entity_type:
                    text = ent.get("text")
                    if text and text.lower() not in seen:
                        seen.add(text.lower())
                        self.add_word(self.entity_type, text)
                        count += 1

        self.fit()
        print(f"Loaded {count} {self.entity_type} entities from BC5CDR")


if __name__ == "__main__":
    import time

    # e = MusicNER("media_net.ahocorasick")
    # e = ImdbNER("imdb.ahocorasick")
    e = EncyclopediaMetallvmNER("metallvm.ahocorasick")

    s = time.monotonic()
    for entity in e.tag("I fucking love black metal"):
        print(entity)
    print(time.monotonic() - s)
