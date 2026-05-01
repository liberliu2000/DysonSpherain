from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from dysonspherain.utils.token_counter import TokenCounter


def calibrate(input_path: Path, output_path: Path, *, strategy: str = "mixed_content_heuristic") -> dict[str, Any]:
    counter = TokenCounter(strategy=strategy)
    rows: list[dict[str, Any]] = []
    by_kind: dict[str, list[float]] = defaultdict(list)
    missing_reference = 0
    if input_path.exists():
        lines = input_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        payload = json.loads(line)
        text = str(payload.get("text") or "")
        kind = str(payload.get("kind") or TokenCounter.classify_text(text))
        count = counter.count(text)
        reference = payload.get("reference_tokens")
        row = {
            "index": index,
            "kind": kind,
            "estimated_tokens": count.tokens,
            "reference_tokens": reference,
            "tokenizer_name": count.tokenizer_name,
            "fallback_tokenizer_used": count.fallback_used,
        }
        if reference is None:
            missing_reference += 1
        else:
            try:
                ref = max(1.0, float(reference))
                if count.tokens > 0:
                    by_kind[kind].append(ref / count.tokens)
                row["relative_error"] = (count.tokens - ref) / ref
            except (TypeError, ValueError):
                missing_reference += 1
        rows.append(row)
    correction_factors = {kind: sum(values) / len(values) for kind, values in by_kind.items() if values}
    if correction_factors:
        correction_factors["default"] = sum(correction_factors.values()) / len(correction_factors)
    payload = {
        "status": "ok",
        "input": str(input_path),
        "sample_count": len(rows),
        "missing_reference_count": missing_reference,
        "strategy": strategy,
        "correction_factors": correction_factors,
        "kind_distribution": {kind: sum(1 for row in rows if row["kind"] == kind) for kind in sorted({row["kind"] for row in rows})},
        "samples": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strategy", default="mixed_content_heuristic")
    args = parser.parse_args(argv)
    payload = calibrate(Path(args.input), Path(args.output), strategy=args.strategy)
    print(json.dumps({key: payload[key] for key in ("status", "sample_count", "missing_reference_count", "correction_factors")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
