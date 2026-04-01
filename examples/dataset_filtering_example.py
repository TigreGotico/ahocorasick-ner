#!/usr/bin/env python
"""Example: Advanced dataset filtering for selective entity extraction.

Demonstrates how to filter datasets by multiple criteria:
- Metal bands by country, genre, and formation year range
- Music tracks by genre, popularity, and audio features
- Movie actors by gender
- Recipe ingredients by source
- Food products by allergens and dietary labels

Requirements:
    pip install ahocorasick-ner "ahocorasick-ner[datasets]"
"""
from ahocorasick_ner.datasets import (
    MetalArchivesBandsNER,
    MetalArchivesTrackNER,
    SpotifyTracksNER,
    MovieActorNER,
    RecipeIngredientsNER,
    FoodProductsNER,
    GenericHFDatasetNER,
)


def example_metal_bands_by_multiple_filters():
    """Extract metal band names using multiple filters."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Metal Bands with Multiple Filters")
    print("=" * 70)

    # Filter by country only
    print("\nLoading Portuguese metal bands...")
    pt_bands = MetalArchivesBandsNER(origin="Portugal")

    # Filter by country and genre
    print("Loading Swedish black metal bands...")
    se_black = MetalArchivesBandsNER(origin="Sweden", genre="Black Metal")

    # Filter by genre and year range
    print("Loading thrash metal bands from 1980-1995...")
    thrash80s = MetalArchivesBandsNER(genre="Thrash Metal",
                                       formed_year_min=1980,
                                       formed_year_max=1995)

    text = "Moonspell, Bathory, and Metallica"
    print(f"\nInput: {text}")

    print("\nPortuguese bands found:")
    for entity in pt_bands.tag(text):
        print(f"  - {entity['word']}")

    print("\nSwedish black metal bands found:")
    for entity in se_black.tag(text):
        print(f"  - {entity['word']}")

    print("\nThrash metal bands (1980-1995) found:")
    for entity in thrash80s.tag(text):
        print(f"  - {entity['word']}")


def example_spotify_by_audio_features():
    """Extract music entities filtered by genre and audio features."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Spotify Tracks by Genre & Audio Features")
    print("=" * 70)

    # Filter by genre
    print("\nLoading rock tracks...")
    rock_tracks = SpotifyTracksNER(genre="rock")

    # Filter by genre and popularity
    print("Loading popular rock tracks (>70 popularity)...")
    popular_rock = SpotifyTracksNER(genre="rock", popularity_min=70)

    # Filter by dance-ability
    print("Loading danceable tracks (danceability > 0.7)...")
    dance = SpotifyTracksNER(danceability_min=0.7)

    # Filter by mood/energy
    print("Loading happy, high-energy tracks...")
    happy = SpotifyTracksNER(energy_min=0.8, valence_min=0.7)

    # Filter by audio characteristics
    print("Loading acoustic instrumental tracks...")
    acoustic = SpotifyTracksNER(acousticness_min=0.8)

    # Filter by explicit content
    print("Loading clean tracks only...")
    clean = SpotifyTracksNER(explicit=False)

    text = "I love Bohemian Rhapsody by Queen and Blinding Lights by The Weeknd"
    print(f"\nInput: {text}")

    print("\nRock tracks found:")
    for entity in rock_tracks.tag(text):
        print(f"  - {entity['word']}")

    print("\nDanceable tracks found:")
    for entity in dance.tag(text):
        print(f"  - {entity['word']}")


def example_generic_filtering():
    """Using GenericHFDatasetNER with custom filtering."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Generic Filtering for Custom Datasets")
    print("=" * 70)

    # Example 1: Load all bands
    print("\nLoading all metal bands (generic)...")
    all_bands = GenericHFDatasetNER(
        entity_type="MetalBand",
        hf_dataset="TigreGotico/metal-archives-bands",
        column="name"
    )

    # Example 2: Load only bands from a specific country
    print("Loading Brazilian metal bands (generic)...")
    brazil_bands = GenericHFDatasetNER(
        entity_type="MetalBand",
        hf_dataset="TigreGotico/metal-archives-bands",
        column="name",
        filter_column="origin",
        filter_value="Brazil"
    )

    text = "Sepultura and Soulfly are from Brazil"
    print(f"\nInput: {text}")

    print("\nBrazilian bands found:")
    for entity in brazil_bands.tag(text):
        print(f"  - {entity['word']}")


def example_movie_actors_by_gender():
    """Filter movie actors by gender."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Movie Actors by Gender")
    print("=" * 70)

    print("\nLoading female actors (actresses)...")
    actresses = MovieActorNER(gender="Female")

    print("Loading male actors...")
    actors = MovieActorNER(gender="Male")

    text = "Tom Hanks, Meryl Streep, and Samuel L. Jackson starred together"
    print(f"\nInput: {text}")

    print("\nFemale actors found:")
    for entity in actresses.tag(text):
        print(f"  - {entity['word']}")

    print("\nMale actors found:")
    for entity in actors.tag(text):
        print(f"  - {entity['word']}")


def example_recipes_and_food_products():
    """Filter recipes and food products by dietary restrictions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Recipes & Food Products by Dietary Restrictions")
    print("=" * 70)

    # Recipe source filtering
    print("\nLoading recipe ingredients from Gathered recipes...")
    gathered_recipes = RecipeIngredientsNER(source="Gathered")

    # Food product filtering
    print("Loading vegan food products...")
    vegan_products = FoodProductsNER(vegan=True)

    print("Loading gluten-free products...")
    gluten_free = FoodProductsNER(allergen="gluten-free")

    print("Loading organic vegetarian products...")
    organic_veg = FoodProductsNER(organic=True, vegetarian=True)

    text = "I bought flour, eggs, and Coca-Cola. Also some tofu and organic milk."
    print(f"\nInput: {text}")

    print("\nRecipe ingredients (from Gathered) found:")
    for entity in gathered_recipes.tag(text):
        print(f"  - {entity['word']}")

    print("\nVegan products found:")
    for entity in vegan_products.tag(text):
        print(f"  - {entity['word']}")

    print("\nGluten-free products found:")
    for entity in gluten_free.tag(text):
        print(f"  - {entity['word']}")


def example_filtering_benefits():
    """Demonstrate memory/performance benefits of filtering."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Filtering Benefits")
    print("=" * 70)

    print("""
Filtering datasets provides several benefits:

1. **Smaller automaton size**: Only entities matching the filter are loaded.
   - All Spotify tracks: 200k+ entities → huge automaton
   - Rock tracks only: Much smaller, faster matching
   - Memory savings: 50-90% reduction possible

2. **Faster matching**: Smaller automaton = fewer comparisons
   - Less trie traversal
   - Better cache locality
   - Faster tag() operations

3. **Semantic clarity**: Entity type matches your domain
   - Metal band detector only finds metal bands (no pop artists)
   - Rock track detector finds rock, not all 200k tracks
   - Reduces false positives in domain-specific applications

4. **Practical for production**: Load only what you need
   - API service: Filter by language, region, or category
   - Mobile app: Load only relevant entity types
   - Embedded systems: Minimal memory footprint
    """)


if __name__ == "__main__":
    print("\n🔍 Dataset Filtering Examples\n")

    try:
        example_metal_bands_by_multiple_filters()
    except Exception as e:
        print(f"⚠️  Example 1 skipped: {e}")

    try:
        example_spotify_by_audio_features()
    except Exception as e:
        print(f"⚠️  Example 2 skipped: {e}")

    try:
        example_generic_filtering()
    except Exception as e:
        print(f"⚠️  Example 3 skipped: {e}")

    try:
        example_movie_actors_by_gender()
    except Exception as e:
        print(f"⚠️  Example 4 skipped: {e}")

    try:
        example_recipes_and_food_products()
    except Exception as e:
        print(f"⚠️  Example 5 skipped: {e}")

    try:
        example_filtering_benefits()
    except Exception as e:
        print(f"⚠️  Example 6 skipped: {e}")

    print("\n✅ Examples completed!\n")
