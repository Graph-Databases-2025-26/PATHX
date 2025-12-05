"""
Extended GALOIS test.

Runs two queries to exercise different branches:

1) WORLD.EU_Cities 
2) GEO.states  

For each query we run all four variants:
Galois_WO, Galois_A, Galois_S, Galois_F
and print comparison tables in the same style.
"""

from src.galois.run_galois import GaloisRunner


def run_eu_cities(runner: GaloisRunner) -> None:
    print("\n" + "=" * 70)
    print("EU_CITIES QUERY (multi-condition WHERE)")
    print("=" * 70)

    results = runner.run_query(
        dataset="WORLD",
        table="EU_Cities",
        attributes=["name", "country", "population", "num_private_hospitals"],
        where_clause="population > 1M AND num_private_hospitals > 15",
        variants_to_run=["Galois_WO", "Galois_A", "Galois_S", "Galois_F"],
    )

    runner.compare_results(results)
    runner.save_results(results, "src/galois/galois_results_eu_cities.json")


def run_geo_states(runner: GaloisRunner) -> None:
    print("\n" + "=" * 70)
    print("GEO.STATES QUERY (single-condition WHERE)")
    print("=" * 70)

    results = runner.run_query(
        dataset="GEO",
        table="states",
        attributes=["name", "area", "population", "capital"],
        where_clause="area > 300000",
        variants_to_run=["Galois_WO", "Galois_A", "Galois_S", "Galois_F"],
    )

    runner.compare_results(results)
    runner.save_results(results, "src/galois/galois_results_geo_states.json")


if __name__ == "__main__":
    runner = GaloisRunner()

    run_eu_cities(runner)
    # run_geo_states(runner)
