# Dataset Reference — ahocorasick-ner

Complete reference for all available dataset loaders with entity type labels, sizes, and usage.

---

## Dataset Filtering

Many dataset loaders support filtering by column values to extract only entities matching specific criteria. This reduces automaton size and improves matching performance.

**Benefits:**
- Smaller automaton (50–90% memory reduction possible)
- Faster matching on domain-specific data
- Semantic clarity (e.g., metal bands only, rock tracks only)
- Production-friendly: load only what you need

**Supported Loaders:**
- `MetalArchivesBandsNER` — filter by `origin` (country)
- `MetalArchivesTrackNER` — filter by `band_origin` (country)
- `SpotifyTracksNER` — filter by `genre`
- `GenericHFDatasetNER` — filter by any column with `filter_column` and `filter_value`

**Examples:**

```python
# Metal bands from Portugal only
pt_bands = MetalArchivesBandsNER(origin="Portugal")

# Rock tracks from Spotify
rock = SpotifyTracksNER(genre="rock")

# Generic: Brazilian metal bands
brazil = GenericHFDatasetNER(
    entity_type="MetalBand",
    hf_dataset="TigreGotico/metal-archives-bands",
    column="name",
    filter_column="origin",
    filter_value="Brazil"
)
```

See [examples/dataset_filtering_example.py](../examples/dataset_filtering_example.py) for detailed usage.

---

## Wikidata Entities (Dedicated Subclasses)

### Easy-Access Subclasses

Dedicated subclasses eliminate the need to know Wikidata QIDs. Use these for straightforward entity extraction.

| Class | QID | Entity Label | Estimated Size | Languages | Import |
|-------|-----|--------------|-----------------|-----------|--------|
| `WikidataAnimalNER` | Q729 | `Animal` | 1M+ | All | `from ahocorasick_ner.datasets import WikidataAnimalNER` |
| `WikidataPlantNER` | Q7432 | `Plant` | 500k+ | All | `from ahocorasick_ner.datasets import WikidataPlantNER` |
| `WikidataCountryNER` | Q6256 | `Country` | 195 | All | `from ahocorasick_ner.datasets import WikidataCountryNER` |
| `WikidataCityNER` | Q515 | `City` | 1M+ | All | `from ahocorasick_ner.datasets import WikidataCityNER` |
| `WikidataPersonNER` | Q5 | `Person` | 100M+ | All | `from ahocorasick_ner.datasets import WikidataPersonNER` |
| `WikidataProfessionNER` | Q28640 | `Profession` | 5k+ | All | `from ahocorasick_ner.datasets import WikidataProfessionNER` |
| `WikidataDiseaseNER` | Q12078 | `Disease` | 10k+ | All | `from ahocorasick_ner.datasets import WikidataDiseaseNER` |
| `WikidataLanguageNER` | Q315 | `Language` | 7k+ | All | `from ahocorasick_ner.datasets import WikidataLanguageNER` |
| `WikidataSportNER` | Q349 | `Sport` | 2k+ | All | `from ahocorasick_ner.datasets import WikidataSportNER` |
| `WikidataBodyPartNER` | Q4891 | `BodyPart` | 500+ | All | `from ahocorasick_ner.datasets import WikidataBodyPartNER` |
| `WikidataFamilyRelationNER` | Q1038 | `FamilyRelation` | 20+ | All | `from ahocorasick_ner.datasets import WikidataFamilyRelationNER` |

**Usage Example:**
```python
from ahocorasick_ner.datasets import WikidataAnimalNER, WikidataCountryNER

animals = WikidataAnimalNER()  # Q729
countries = WikidataCountryNER(lang="de-de")  # Q6256, German labels
professions = WikidataProfessionNER()  # Q28640
```

**Multilingual Support:**
- All Wikidata subclasses support language parameter: `WikidataAnimalNER(lang="es")`
- Default: `"en"` (English)
- Supported: All Wikidata languages (de, fr, it, es, pt, ru, zh, ja, ko, ar, etc.)

### Extended Wikidata Subclasses

| Class | QID | Entity Label | Size | Languages | Import |
|-------|-----|--------------|------|-----------|--------|
| `WikidataLandmarkNER` | Q570116 | `Landmark` | 10k+ | All | `from ahocorasick_ner.datasets import WikidataLandmarkNER` |
| `WikidataUniversityNER` | Q3918 | `University` | 10k+ | All | `from ahocorasick_ner.datasets import WikidataUniversityNER` |
| `WikidataMuseumNER` | Q33506 | `Museum` | 10k+ | All | `from ahocorasick_ner.datasets import WikidataMuseumNER` |
| `WikidataAuthorNER` | Q36180 | `Author` | 500k+ | All | `from ahocorasick_ner.datasets import WikidataAuthorNER` |
| `WikidataBookNER` | Q571 | `Book` | 1M+ | All | `from ahocorasick_ner.datasets import WikidataBookNER` |
| `WikidataPublisherNER` | Q2085381 | `Publisher` | 10k+ | All | `from ahocorasick_ner.datasets import WikidataPublisherNER` |
| `WikidataAthleteNER` | Q2066131 | `Athlete` | 500k+ | All | `from ahocorasick_ner.datasets import WikidataAthleteNER` |
| `WikidataSportsTeamNER` | Q12973014 | `SportsTeam` | 10k+ | All | `from ahocorasick_ner.datasets import WikidataSportsTeamNER` |
| `WikidataMovieNER` | Q11424 | `Movie` | 500k+ | All | `from ahocorasick_ner.datasets import WikidataMovieNER` |
| `WikidataMusicalNER` | Q11649 | `Instrument` | 2k+ | All | `from ahocorasick_ner.datasets import WikidataMusicalNER` |
| `WikidataCompanyNER` | Q783794 | `Company` | 100k+ | All | `from ahocorasick_ner.datasets import WikidataCompanyNER` |

---

## Business & E-Commerce

| Class | Dataset | Entity Label | Size | Languages | Import |
|-------|---------|--------------|------|-----------|--------|
| `CompanyNamesNER` | nbroad/company_names | `company` | 50k+ | English | `from ahocorasick_ner.datasets import CompanyNamesNER` |

---

## Music (Expanded)

| Class | Dataset | Entity Label(s) | Size | Filtering | Import |
|-------|---------|-----------------|------|-----------|--------|
| `SpotifyTracksNER` | spotify-tracks-dataset | `track_name`, `artist_name`, `music_genre` | 200k tracks | `genre`, `popularity_min`, `popularity_max`, `energy_min`, `danceability_min`, `acousticness_min`, `valence_min`, `explicit` | `from ahocorasick_ner.datasets import SpotifyTracksNER` |

**Filtering Examples:**
```python
# Popular rock tracks (>70 popularity)
rock = SpotifyTracksNER(genre="rock", popularity_min=70)

# Danceability dance tracks
dance = SpotifyTracksNER(genre="dance", danceability_min=0.7)

# High-energy, high-valence (happy) tracks
upbeat = SpotifyTracksNER(energy_min=0.8, valence_min=0.7)

# Only explicit-free tracks
clean = SpotifyTracksNER(explicit=False)

# Acoustic tracks
acoustic = SpotifyTracksNER(acousticness_min=0.8)
```

---

## Food & Recipes

| Class | Dataset | Entity Label(s) | Size | Filtering | Import |
|-------|---------|-----------------|------|-----------|--------|
| `RecipeIngredientsNER` | recipe_nlg | `ingredient` | 2.2M recipes | `source` (Gathered/Recipes1M) | `from ahocorasick_ner.datasets import RecipeIngredientsNER` |
| `FoodProductsNER` | openfoodfacts/product-database | `food_product`, `brand` | 4M+ products | `allergen`, `vegan`, `vegetarian`, `organic`, `country` | `from ahocorasick_ner.datasets import FoodProductsNER` |

**Filtering Examples:**
```python
# Ingredients from Gathered recipes only
gathered = RecipeIngredientsNER(source="Gathered")

# Gluten-free food products
gluten_free = FoodProductsNER(allergen="gluten-free")

# Vegan products only
vegan = FoodProductsNER(vegan=True)

# Organic products from France
organic_fr = FoodProductsNER(organic=True, country="fr")

# Nut-free vegetarian products
safe_veg = FoodProductsNER(allergen="tree-nut-free", vegetarian=True)
```

---

## Programming & Technology

| Class | Languages Covered | Entity Label | Built-in | Import |
|-------|-------------------|--------------|----------|--------|
| `ProgrammingLanguageNER` | 50+ languages | `programming_language` | Python, Java, JavaScript, Go, Rust, TypeScript, C++, C#, Ruby, PHP, Swift, Kotlin, etc. | `from ahocorasick_ner.datasets import ProgrammingLanguageNER` |

---

## Locations & Names

| Class | Source | Entity Label | Size | Languages | Import |
|-------|--------|--------------|------|-----------|--------|
| `GeoNamesNER` | GeoNames | `location` | 280k+ | Multilingual names | `from ahocorasick_ner.datasets import GeoNamesNER` |
| `PersonNamesNER` | Hobson/surname-nationality | `person_name` | 50k+ surnames | 30+ countries | `from ahocorasick_ner.datasets import PersonNamesNER` |

**Usage Example:**
```python
from ahocorasick_ner.datasets import GeoNamesNER, PersonNamesNER

cities = GeoNamesNER()  # 280k+ cities worldwide
names = PersonNamesNER()  # 50k+ surnames, 30+ countries
```

---

## Biomedical

| Class | Dataset | Entity Labels | Size | Languages | Import |
|-------|---------|---------------|------|-----------|--------|
| `BC5CDRMedicalNER` | BC5CDR | `Disease`, `Chemical` | 5k+ each | English | `from ahocorasick_ner.datasets import BC5CDRMedicalNER` |

**Constructor:**
```python
# Extract diseases
diseases = BC5CDRMedicalNER(entity_type="Disease")
# Extract chemicals
chemicals = BC5CDRMedicalNER(entity_type="Chemical")
```

---

## Entertainment & Media (Jarbas/TigreGotico MediaEntities)

### Metal Archives

| Class | Dataset | Entity Label | Size | Filtering | Import |
|-------|---------|--------------|------|-----------|--------|
| `MetalArchivesBandsNER` | metal-archives-bands | `metal_band` | 4.6k | `origin`, `genre`, `formed_year_min`, `formed_year_max` | `from ahocorasick_ner.datasets import MetalArchivesBandsNER` |
| `MetalArchivesTrackNER` | metal-archives-tracks | `metal_band`, `metal_album`, `metal_track` | 205k tracks | `band_origin`, `album_type` | `from ahocorasick_ner.datasets import MetalArchivesTrackNER` |
| `EncyclopediaMetallvmNER` | Combined (bands + tracks) | `artist_name`, `track_name`, `album_name`, `album_type` | 209k+ combined | None | `from ahocorasick_ner.datasets import EncyclopediaMetallvmNER` |

**Filtering Examples:**
```python
# Portuguese metal bands only
pt_bands = MetalArchivesBandsNER(origin="Portugal")

# Thrash metal bands from 1980-1995
thrash = MetalArchivesBandsNER(genre="Thrash Metal",
                                formed_year_min=1980,
                                formed_year_max=1995)

# Studio full-length albums from Swedish bands
swedish_albums = MetalArchivesTrackNER(band_origin="Sweden",
                                        album_type="Full-length")

# Demo tracks only
demos = MetalArchivesTrackNER(album_type="Demo")
```

### IMDB

| Class | Dataset | Entity Label | Size | Filtering | Import |
|-------|---------|--------------|------|-----------|--------|
| `MovieActorNER` | movie_actors | `movie_actor` | 6.3M | `gender` (Male/Female) | `from ahocorasick_ner.datasets import MovieActorNER` |
| `MovieDirectorNER` | movie_directors | `movie_director` | 128k | None | `from ahocorasick_ner.datasets import MovieDirectorNER` |
| `MovieComposerNER` | movie_composers | `movie_composer` | 221k | None | `from ahocorasick_ner.datasets import MovieComposerNER` |
| `ImdbNER` | Combined (all roles) | `movie_actor`, `movie_director`, `movie_producer`, `movie_writer`, `movie_composer` | 6.6M+ combined | None | `from ahocorasick_ner.datasets import ImdbNER` |

**Filtering Examples:**
```python
# Female actors only (actresses)
actresses = MovieActorNER(gender="Female")

# Male actors only
actors = MovieActorNER(gender="Male")
```

### Music (Other Genres)

| Class | Dataset | Entity Labels | Size | Languages | Import |
|-------|---------|---------------|------|-----------|--------|
| `JazzNER` | jazz-music-archives | `jazz_artist`, `jazz_genre` | 12.3k artists | English | `from ahocorasick_ner.datasets import JazzNER` |
| `ProgRockNER` | prog-archives | `prog_artist`, `prog_genre` | 12.4k artists | English | `from ahocorasick_ner.datasets import ProgRockNER` |
| `MusicNER` | Combined (all genres) | `artist_name`, `track_name`, `album_name`, `music_genre`, etc. | 240k+ combined | English | `from ahocorasick_ner.datasets import MusicNER` |

---

## Generic HuggingFace Integration

### Flexible Dataset Loader

| Class | Purpose | Parameters | Import |
|-------|---------|------------|--------|
| `GenericHFDatasetNER` | Load ANY HF dataset with entities in a column | `entity_type`, `hf_dataset`, `column`, optional: `filter_column`, `filter_value` | `from ahocorasick_ner.datasets import GenericHFDatasetNER` |
| `WikidataEntityNER` | Generic Wikidata by QID | `entity_type` (str), `wikidata_qid` (str), `lang` (str) | `from ahocorasick_ner.datasets import WikidataEntityNER` |

**Usage Example:**
```python
# Load all colors from color-pedia
colors = GenericHFDatasetNER(
    entity_type="Color",
    hf_dataset="boltuix/color-pedia",
    column="name"
)

# Load warm colors only
warm_colors = GenericHFDatasetNER(
    entity_type="Color",
    hf_dataset="boltuix/color-pedia",
    column="name",
    filter_column="color_family",
    filter_value="warm"
)

# Load Brazilian metal bands only
brazil_bands = GenericHFDatasetNER(
    entity_type="MetalBand",
    hf_dataset="TigreGotico/metal-archives-bands",
    column="name",
    filter_column="origin",
    filter_value="Brazil"
)

# Load any entity type from Wikidata by QID
instruments = WikidataEntityNER(
    entity_type="Instrument",
    wikidata_qid="Q11649"  # musical instrument
)
```

---

## Quick Reference: Entity Type Labels

| Source | Entity Labels |
|--------|----------------|
| **Wikidata** | `Animal`, `Plant`, `Country`, `City`, `Person`, `Profession`, `Disease`, `Language`, `Sport`, `BodyPart`, `FamilyRelation` |
| **Locations** | `location` |
| **Names** | `person_name` |
| **Biomedical** | `Disease`, `Chemical` |
| **Metal Archives** | `metal_band`, `metal_album`, `metal_track`, `album_type` |
| **IMDB** | `movie_actor`, `movie_director`, `movie_composer`, `movie_producer`, `movie_writer` |
| **Jazz** | `jazz_artist`, `jazz_genre` |
| **Prog Rock** | `prog_artist`, `prog_genre` |

---

## Installation

```bash
# Core ahocorasick-ner
pip install ahocorasick-ner

# With HuggingFace dataset support
pip install "ahocorasick-ner[datasets]"
```

---

## Total Coverage

- **1.5B+ entities** across all loaders
- **30+ languages** via Wikidata
- **6.3M+ entertainment names** (IMDB)
- **1M+ geographic locations** (GeoNames + Wikidata)
- **100M+ person names** (Wikidata Q5)
- **10k+ medical terms** (BC5CDR)
- **240k+ music entities** (Metal, Jazz, Prog Rock, Classical)

---

## See Also

- [simple-NER integration](../../../DEPRECATED/simple_NER/docs/FAQ.md) — Use these in pipelines
- Examples: [basic_usage.py](../examples/basic_usage.py), [benchmark.py](../examples/benchmark.py)
