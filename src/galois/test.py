# Test query with full result display
from src.galois.run_galois import GaloisRunner

print("""
╔═══════════════════════════════════════════════════════════╗
║         GALOIS Test Query                                 ║
╚═══════════════════════════════════════════════════════════╝
""")

runner = GaloisRunner()

# Query: European countries with > 60M population AND > 3T GDP
results = runner.run_query(
    dataset="WORLD",
    table="EU_Cities",
    attributes=["name", "country", "population", " # pvt hosps"],
    where_clause="population > 1M AND  # pvt hosp > 15",
    
    # Pour debug plus rapide, décommenter pour tester seulement A et F:
    variants_to_run=["Galois_A","Galois_S", "Galois_F"]
)

print("\n" + "="*70)
print("DISPLAYING RESULTS")
print("="*70)

# Afficher les résultats avec tables
runner.compare_results(results)

# Sauvegarder pour analyse ultérieure
runner.save_results(results, "src/galois/galois_results.json")

print("\n" + "="*70)
print("TIPS:")
print("="*70)
print("1. Les tables sont affichées dans la section 'RESULT SET CONTENT'")
print("2. Pour affichage plus joli: python -m src.galois.print_results galois_results.json")
print("3. Pour debug prompts: python -m src.galois.debug_prompts")
print("4. Voir TROUBLESHOOTING.md si problèmes")
print("="*70 + "\n")