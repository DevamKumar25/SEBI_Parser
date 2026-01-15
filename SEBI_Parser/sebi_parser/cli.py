import json
from .sebi import parse_sebi_pdf

def main():
    results = parse_sebi_pdf()

    if results:
        output_file = "sebi_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 70)
        print(f"Results saved to {output_file}")
        print(f"Relevant items: {sum(len(v) for v in results.values())}")
        print("=" * 70)
    else:
        print("\nNo results")

if __name__ == "__main__":
    main()
