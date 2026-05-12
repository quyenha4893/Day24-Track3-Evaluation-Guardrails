"""CI eval gate: re-run RAGAS, exit 1 if threshold breached."""

import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", action="append", default=[],
                        help="metric=value, vd: faithfulness=0.85")
    parser.add_argument("--summary", default="phase-a/ragas_summary.json")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    thresholds = {}
    for t in args.threshold:
        k, v = t.split("=")
        thresholds[k.strip()] = float(v.strip())

    print(f"Loaded summary: {summary}")
    print(f"Thresholds: {thresholds}")

    failed = []
    for metric, target in thresholds.items():
        actual = summary.get(metric)
        if actual is None:
            failed.append((metric, "missing", target))
        elif actual < target:
            failed.append((metric, actual, target))

    if failed:
        print("EVAL GATE FAILED:")
        for m, a, t in failed:
            print(f"  [FAIL] {m}: actual={a}, target={t}")
        sys.exit(1)
    print("EVAL GATE PASSED")
    sys.exit(0)

if __name__ == "__main__":
    main()
