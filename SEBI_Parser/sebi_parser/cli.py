import json
from .sebi import parse_sebi_pdf
from collections import defaultdict
from datetime import datetime

# def main():
#     results = parse_sebi_pdf()

#     if results:
#         output_file = "sebi_results.json"
#         with open(output_file, "w", encoding="utf-8") as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)

#         print("\n" + "=" * 70)
#         print(f"Results saved to {output_file}")
#         print(f"Relevant items: {sum(len(v) for v in results.values())}")
#         print("=" * 70)
#     else:
#         print("\nNo results")


def main():
    failure = defaultdict(dict)
    success = defaultdict(dict)
    results = parse_sebi_pdf()

    if results:
        for r in results:
            if "error" in r:
                failure[r["error"]] = r
            else:
                pub_date = r["publish_date"]
                success[pub_date][f"item_{counter}"] = r
                counter += 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        if success:
            success_file = f"success_{ts}.json"
            with open(success_file, "w", encoding="utf-8") as f:
                json.dump(dict(success), f, indent=2, ensure_ascii=False)
            print(f"\n✅ Success: {counter - 1} items saved to {success_file}")
        else:
             print("\nℹ️ No successful items to save.")

        if failure:
            failure_file = f"failure_{ts}.json"
            with open(failure_file, "w", encoding="utf-8") as f:
                json.dump(failure, f, indent=2, ensure_ascii=False)
            print(f"❌ Failures: {len(failure)} items saved to {failure_file}")
            
    except Exception as e:
        print(f"\n❌ CRITICAL: Failed to save results to JSON: {e}")
            

    print("\n" + "=" * 70)
    print(f"Success: {len(success)}")
    print(f"Failure: {len(failure)}")
    print("=" * 70)

if __name__ == "__main__":
    main()

