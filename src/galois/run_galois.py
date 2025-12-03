"""
Main script to run and compare all Galois variants
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.baselines.llm_factory import get_llm_wrapper
from src.galois.operators.galois_wo import GaloisWOOperator
from src.galois.variants.galois_a import GaloisA
from src.galois.variants.galois_s import GaloisS
from src.galois.variants.galois_f import GaloisF
from src.galois.env_config import load_config, GaloisConfig


class GaloisRunner:
    """
    Runner to execute and compare all Galois variants.
    """

    def __init__(self, config: Optional[GaloisConfig] = None):
        """
        Args:
            config: GaloisConfig instance. If None, loads from .env
        """
        if config is None:
            config = load_config()
        
        self.config = config
        
        # Get LLM wrapper
        wrapper = get_llm_wrapper(config)
        self.llm = wrapper.get_llm_instance()
        
        # Initialize all variants
        self.variants = {
            "Galois_WO": GaloisWOOperator(self.llm, max_iters=5),
            "Galois_A": GaloisA(self.llm, max_iters=5),
            "Galois_S": GaloisS(self.llm, max_iters=5),
            "Galois_F": GaloisF(
                self.llm, 
                max_iters=5,
                confidence_threshold=0.6,  # Tau from paper
            ),
        }

    def run_query(
        self,
        dataset: str,
        table: str,
        attributes: list[str],
        where_clause: str | None = None,
        variants_to_run: list[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Run query through specified Galois variants.
        
        Args:
            dataset: Dataset name (e.g., "WORLD")
            table: Table name (e.g., "city")
            attributes: List of attribute names
            where_clause: SQL WHERE clause (without "WHERE")
            variants_to_run: List of variant names to run (default: all)
        
        Returns:
            Dict mapping variant_name -> result
        """
        if variants_to_run is None:
            variants_to_run = list(self.variants.keys())
        
        results = {}
        
        print(f"\n{'='*70}")
        print(f"Running Query:")
        print(f"  Dataset: {dataset}")
        print(f"  Table: {table}")
        print(f"  Attributes: {', '.join(attributes)}")
        print(f"  WHERE: {where_clause or '(none)'}")
        print(f"{'='*70}\n")
        
        for variant_name in variants_to_run:
            if variant_name not in self.variants:
                print(f"⚠️  Unknown variant: {variant_name}")
                continue
            
            print(f"🔍 Running {variant_name}...")
            
            try:
                variant = self.variants[variant_name]
                
                # Execute based on variant type
                if isinstance(variant, GaloisWOOperator):
                    result = variant.scan(dataset, table, attributes, where_clause)
                else:
                    result = variant.execute(dataset, table, attributes, where_clause)
                
                results[variant_name] = result
                
                print(f"   ✓ {len(result['result_set'])} rows, "
                      f"{result['tokens']:,} tokens, "
                      f"{result['time']:.2f}s")
            
            except Exception as e:
                print(f"   ✗ Error: {e}")
                results[variant_name] = {"error": str(e)}
        
        return results

    def compare_results(self, results: Dict[str, Any]) -> None:
        """
        Print comparison table of results.
        """
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"{'Variant':<15} {'Rows':<8} {'Tokens':<15} {'Time (s)':<10} {'Strategy':<15}")
        print(f"{'-'*70}")
        
        for variant_name, result in results.items():
            if "error" in result:
                print(f"{variant_name:<15} ERROR: {result['error']}")
                continue
            
            strategy = result.get("meta", {}).get("strategy", "N/A")
            print(f"{variant_name:<15} "
                  f"{len(result['result_set']):<8} "
                  f"{result['tokens']:<15,} "
                  f"{result['time']:<10.2f} "
                  f"{strategy:<15}")
        
        print(f"\n{'='*70}")
        print("RESULT SET CONTENT")
        print(f"{'='*70}")
        
        for variant_name, result in results.items():
            print(f"\n--- {variant_name} ({len(result['result_set'])} rows) ---")
            if "error" in result:
                print(f"  ERROR: {result['error']}")
                continue
            
            result_set = result.get('result_set', [])
            
            # Imprimer l'en-tête (si disponible)
            if result_set:
                headers = list(result_set[0].keys()) if isinstance(result_set[0], dict) else ["Row"]
                print(" | ".join(headers))
                print("-" * (len(" | ".join(headers)) + 2))
            
            # Imprimer les lignes
            for row in result_set:
                if isinstance(row, dict):
                    print(" | ".join(str(v) for v in row.values()))
                else:
                    print(row)


        # Token cost comparison
        baseline_name = "Galois_WO"
        if baseline_name in results and "error" not in results[baseline_name]:
            baseline_tokens = results[baseline_name]["tokens"]
            
            print(f"\n{'='*70}")
            print(f"TOKEN COST RELATIVE TO {baseline_name}:")
            print(f"{'-'*70}")
            
            for variant_name, result in results.items():
                if variant_name == baseline_name or "error" in result:
                    continue
                
                tokens = result["tokens"]
                ratio = tokens / baseline_tokens if baseline_tokens > 0 else 0
                reduction = (1 - ratio) * 100
                
                print(f"  {variant_name:<15}: {ratio:>6.1%} "
                      f"({reduction:>+6.1f}% reduction)")

    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {filepath}")


def example_world_cities(config: Optional[GaloisConfig] = None):
    """
    Example from the paper: European cities query
    
    SELECT name, size, population, num_private_hospitals
    FROM EU_Cities
    WHERE population > 1M AND num_private_hospitals > 15
    """
    if config is None:
        config = load_config()
    
    runner = GaloisRunner(config)
    
    results = runner.run_query(
        dataset="WORLD",
        table="EU_Cities",
        attributes=["name", "country", "population", "num_private_hospitals"],
        where_clause="population > 1000000 AND num_private_hospitals > 15",
    )
    
    runner.compare_results(results)
    return results


def example_geo_states(config: Optional[GaloisConfig] = None):
    """
    Example: Geographic query with single condition
    
    SELECT name, area, population, capital
    FROM states
    WHERE area > 300000
    """
    if config is None:
        config = load_config()
    
    runner = GaloisRunner(config)
    
    results = runner.run_query(
        dataset="GEO",
        table="states",
        attributes=["name", "area", "population", "capital"],
        where_clause="area > 300000",
    )
    
    runner.compare_results(results)
    return results


def example_custom_query(config: Optional[GaloisConfig] = None, 
                        dataset: str = "", table: str = "", 
                        attributes: list = None, where_clause: str = ""):
    """Run custom query through all variants"""
    if config is None:
        config = load_config()
    
    runner = GaloisRunner(config)
    
    results = runner.run_query(
        dataset=dataset,
        table=table,
        attributes=attributes,
        where_clause=where_clause,
    )
    
    runner.compare_results(results)
    return results


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         GALOIS: Query Optimization for LLMs               ║
    ║                                                            ║
    ║  Comparing four execution strategies:                     ║
    ║  • Galois_WO: Cell-by-cell (baseline)                     ║
    ║  • Galois_A:  Push all conditions                         ║
    ║  • Galois_S:  Push selective condition                    ║
    ║  • Galois_F:  Full optimization (confidence-based)        ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("\n⚠️  To run examples:")
    print("   1. Create .env file in project root with your API keys")
    print("   2. Run: python -c 'from src.galois.run_galois import example_world_cities; example_world_cities()'")
    print("   3. Or: from src.galois.run_galois import GaloisRunner")
    print("          runner = GaloisRunner()  # Auto-loads from .env")
    print()