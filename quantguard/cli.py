"""Command-line interface for QuantGuard.

    quantguard verify answer.txt --sources sources.json
    quantguard benchmark dataset.json

Uses only the standard library (argparse) -- the core library has no
required dependencies, and the CLI shouldn't add one just to exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from quantguard import QuantitativeGuard
from quantguard.models import Status


def _load_sources(path: Path) -> list[str]:
    """Sources file is either a JSON list of strings, or a JSON object
    with a "sources" or "chunks" key holding that list.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict):
        for key in ("sources", "chunks", "source_chunks"):
            if key in data:
                return [str(item) for item in data[key]]
    raise ValueError(f"Could not find a list of source chunks in {path}")


def _print_human(result, verbose: bool) -> None:
    print(f"Overall valid: {result.is_valid}")
    print(f"Status counts: {result.status_counts}")
    print()
    for r in result.results:
        print(f"[{r.status.value}] {r.claim.raw_text!r}")
        print(f"    {r.reason}")
        if verbose:
            for line in r.trace:
                print(f"    - {line}")
    if not result.is_valid:
        patched, patches = result.auto_correct()
        if patches:
            print()
            print("Suggested corrections:")
            for p in patches:
                print(f"  {p.original!r} -> {p.replacement!r}  ({p.reason}, confidence {p.confidence:.2f})")
            print()
            print("Patched text:")
            print(patched)


def cmd_verify(args: argparse.Namespace) -> int:
    answer_path = Path(args.answer_file)
    text = answer_path.read_text(encoding="utf-8")
    sources = _load_sources(Path(args.sources))

    guard = QuantitativeGuard(tolerance=args.tolerance)
    result = guard.verify(text, sources)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print_human(result, verbose=args.verbose)

    if args.fail_on_invalid and not result.is_valid:
        return 1
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    guard = QuantitativeGuard(tolerance=args.tolerance)

    total = len(dataset)
    correct = 0
    for case in dataset:
        result = guard.verify(case["generated_text"], case["source_chunks"])
        ok = result.is_valid == case["expected_valid"]
        correct += ok
        mark = "OK" if ok else "WRONG"
        print(f"[{mark}] {case.get('id', '?'):30} expected={case['expected_valid']!s:5} got={result.is_valid!s:5}")

    print()
    print(f"Accuracy: {correct}/{total} = {correct / total:.1%}" if total else "No cases in dataset.")
    return 0 if correct == total else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantguard", description="Deterministic claim verification for RAG answers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify", help="Verify a generated answer against source chunks.")
    verify_parser.add_argument("answer_file", help="Path to a text file containing the generated answer.")
    verify_parser.add_argument("--sources", required=True, help="Path to a JSON file: a list of strings, or {\"sources\": [...]}.")
    verify_parser.add_argument("--tolerance", type=float, default=0.05, help="Relative verification tolerance (default 0.05).")
    verify_parser.add_argument("--json", action="store_true", help="Print the full result as JSON instead of a human-readable summary.")
    verify_parser.add_argument("--verbose", action="store_true", help="Include the per-claim trace in human-readable output.")
    verify_parser.add_argument("--fail-on-invalid", action="store_true", help="Exit with status 1 if any claim is not valid.")
    verify_parser.set_defaults(func=cmd_verify)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run QuantGuard over a labeled dataset and report accuracy.")
    benchmark_parser.add_argument("dataset", help="Path to a JSON file: a list of {generated_text, source_chunks, expected_valid}.")
    benchmark_parser.add_argument("--tolerance", type=float, default=0.05)
    benchmark_parser.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
