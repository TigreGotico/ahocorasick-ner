"""Benchmark comparing all three backends on a city-names dataset."""
import time
from typing import List, Tuple

from ahocorasick_ner import AhocorasickNER
from ahocorasick_ner.numpy_backend import NumpyAhocorasickNER

# --- Dataset: 200 world cities ---
CITIES: List[Tuple[str, str]] = [
    ("city", c) for c in [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
        "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
        "Indianapolis", "San Francisco", "Seattle", "Denver", "Washington",
        "Nashville", "Oklahoma City", "El Paso", "Boston", "Portland",
        "Las Vegas", "Memphis", "Louisville", "Baltimore", "Milwaukee",
        "Albuquerque", "Tucson", "Fresno", "Sacramento", "Mesa",
        "Kansas City", "Atlanta", "Omaha", "Colorado Springs", "Raleigh",
        "Long Beach", "Virginia Beach", "Miami", "Oakland", "Minneapolis",
        "Tampa", "Tulsa", "Arlington", "New Orleans", "Cleveland",
        "London", "Birmingham", "Manchester", "Leeds", "Glasgow",
        "Liverpool", "Edinburgh", "Bristol", "Sheffield", "Leicester",
        "Paris", "Marseille", "Lyon", "Toulouse", "Nice",
        "Nantes", "Strasbourg", "Montpellier", "Bordeaux", "Lille",
        "Tokyo", "Yokohama", "Osaka", "Nagoya", "Sapporo",
        "Fukuoka", "Kawasaki", "Kobe", "Kyoto", "Saitama",
        "Berlin", "Hamburg", "Munich", "Cologne", "Frankfurt",
        "Stuttgart", "Dusseldorf", "Leipzig", "Dortmund", "Essen",
        "Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide",
        "Canberra", "Hobart", "Darwin", "Newcastle", "Gold Coast",
        "Toronto", "Montreal", "Vancouver", "Calgary", "Edmonton",
        "Ottawa", "Winnipeg", "Quebec City", "Hamilton", "Halifax",
        "Mexico City", "Guadalajara", "Monterrey", "Puebla", "Tijuana",
        "Cancun", "Merida", "Leon", "Toluca", "Queretaro",
        "Buenos Aires", "Cordoba", "Rosario", "Mendoza", "La Plata",
        "Tucuman", "Mar del Plata", "Salta", "Santa Fe", "Neuquen",
        "Sao Paulo", "Rio de Janeiro", "Brasilia", "Salvador", "Fortaleza",
        "Belo Horizonte", "Manaus", "Curitiba", "Recife", "Porto Alegre",
        "Moscow", "Saint Petersburg", "Novosibirsk", "Yekaterinburg", "Kazan",
        "Chelyabinsk", "Samara", "Omsk", "Rostov-on-Don", "Ufa",
        "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad",
        "Chennai", "Kolkata", "Pune", "Jaipur", "Lucknow",
        "Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Chengdu",
        "Wuhan", "Hangzhou", "Nanjing", "Chongqing", "Tianjin",
        "Cairo", "Lagos", "Kinshasa", "Johannesburg", "Nairobi",
        "Casablanca", "Addis Ababa", "Dar es Salaam", "Accra", "Cape Town",
    ]
]

# --- Sentences to tag ---
SENTENCES = [
    "I flew from New York to London and then took a train to Paris.",
    "The conference in Tokyo attracted researchers from Beijing, Seoul, and Mumbai.",
    "She grew up in Buenos Aires but moved to San Francisco for a tech job.",
    "Berlin and Munich are popular destinations in Germany.",
    "The route goes through Cairo, Nairobi, and Cape Town.",
    "Sao Paulo and Rio de Janeiro are the largest cities in Brazil.",
    "From Moscow to Saint Petersburg is an overnight train ride.",
    "Sydney and Melbourne compete for the title of best Australian city.",
    "The flight from Toronto to Vancouver takes about five hours.",
    "Mexico City is one of the most populous cities in the world.",
] * 100  # repeat for more stable timing


def bench(name: str, ner_instance, sentences: list) -> None:
    """Run tagging benchmark and print results."""
    # Warmup
    for s in sentences[:10]:
        list(ner_instance.tag(s, min_word_len=3))

    start = time.perf_counter()
    total_tags = 0
    total_chars = 0
    for s in sentences:
        tags = list(ner_instance.tag(s, min_word_len=3))
        total_tags += len(tags)
        total_chars += len(s)
    elapsed = time.perf_counter() - start

    print(f"{name:>20}: {elapsed*1000:8.1f}ms | "
          f"{total_chars:,} chars | "
          f"{total_tags:,} tags | "
          f"{total_chars/elapsed:,.0f} chars/sec")


def main() -> None:
    """Run benchmarks across all backends."""
    # Build pyahocorasick backend
    ner_c = AhocorasickNER()
    for label, city in CITIES:
        ner_c.add_word(label, city)
    ner_c.fit()

    # Build numpy backend
    ner_np = NumpyAhocorasickNER()
    for label, city in CITIES:
        ner_np.add_word(label, city)
    ner_np.fit()

    # ONNX backend (optional)
    try:
        from ahocorasick_ner.onnx_backend import OnnxAhocorasickNER
        ner_onnx = OnnxAhocorasickNER()
        for label, city in CITIES:
            ner_onnx.add_word(label, city)
        ner_onnx.fit()
        has_onnx = True
    except ImportError:
        has_onnx = False

    print(f"Dataset: {len(CITIES)} cities, {len(SENTENCES)} sentences")
    print("-" * 75)

    bench("pyahocorasick (C)", ner_c, SENTENCES)
    bench("numpy (pure Python)", ner_np, SENTENCES)
    if has_onnx:
        bench("onnx (onnxruntime)", ner_onnx, SENTENCES)


if __name__ == "__main__":
    main()
