#!/usr/bin/env python3
"""Generate LaTeX/PGF paper figures from artifact-backed tables."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "latex" / "figures"
FORMAL_JSON = ROOT / "artifacts" / "formal_protocol_validation.json"
MAIN_RESULTS_TEX = ROOT / "paper" / "tables" / "main_results.tex"
PARETO_CSV = ROOT / "paper" / "figures" / "data" / "pareto_curve_data.csv"


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n")


def pipeline() -> None:
    write(
        FIG_DIR / "pipeline_figure.tex",
        r"""\begin{tikzpicture}[
  node distance=7mm and 9mm,
  box/.style={draw=gray!65, rounded corners=2pt, align=center, minimum width=36mm, minimum height=8mm, font=\small, fill=gray!4},
  chan/.style={draw=gray!65, rounded corners=2pt, align=center, minimum width=26mm, minimum height=7mm, font=\scriptsize, fill=blue!4},
  fail/.style={font=\scriptsize, text=red!70!black},
  arrow/.style={-Latex, thick, draw=gray!70}
]
\node[box] (query) {Query};
\node[box, below=of query] (route) {Route Classifier};
\node[box, below=of route] (admit) {Multi-channel Candidate Admission};
\node[chan, below left=of admit] (dense) {Dense};
\node[chan, below=of admit] (lex) {Lexical / Exact};
\node[chan, below right=of admit] (temporal) {Temporal};
\node[chan, below=of lex, xshift=-24mm] (entity) {Entity / Profile};
\node[chan, below=of lex, xshift=24mm] (parent) {Parent / Session};
\node[box, below=22mm of lex] (fusion) {Dense-Preserving Safe Fusion};
\node[box, below=of fusion] (segment) {Parent-to-Segment Expansion};
\node[box, below=of segment] (rerank) {Reranking};
\node[box, below=of rerank] (out) {Evidence Context + Diagnostics};

\draw[arrow] (query) -- (route);
\draw[arrow] (route) -- node[fail, right] {F2 route miss} (admit);
\draw[arrow] (admit) -- (dense);
\draw[arrow] (admit) -- (lex);
\draw[arrow] (admit) -- (temporal);
\draw[arrow] (admit) -- (entity);
\draw[arrow] (admit) -- (parent);
\draw[arrow] (dense) -- node[fail, left] {F1 dense miss} (fusion);
\draw[arrow] (lex) -- (fusion);
\draw[arrow] (temporal) -- (fusion);
\draw[arrow] (entity) -- (fusion);
\draw[arrow] (parent) -- (fusion);
\draw[arrow] (fusion) -- node[fail, right] {F5 fusion downrank} (segment);
\draw[arrow] (segment) -- node[fail, right] {F3 parent hit / segment miss} (rerank);
\draw[arrow] (rerank) -- node[fail, right] {F4 local crowding} (out);
\end{tikzpicture}""",
    )


def parse_metric_blob(blob: str) -> dict[str, float]:
    metrics = {}
    for key, value in re.findall(r"([A-Za-z_@0-9]+)=([0-9.]+)", blob):
        metrics[key] = float(value)
    return metrics


def baseline_rows() -> list[dict[str, object]]:
    rows = []
    pattern = re.compile(r"^([^&]+)&([^&]+)&([^&]+)&(.+)\\\\$")
    for line in MAIN_RESULTS_TEX.read_text().splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        benchmark, method, status, blob = [part.strip() for part in match.groups()]
        if benchmark == "Benchmark" or status != "available":
            continue
        if method not in {"bm25", "dense_only_minilm", "dense_bm25_rrf", "dysonspherain_full"}:
            continue
        rows.append({"benchmark": benchmark, "method": method, "metrics": parse_metric_blob(blob)})
    return rows


def recall(metrics: dict[str, float]) -> float | None:
    for key in ("final_recall@10", "recall_frac@10", "recall_any@10"):
        if key in metrics:
            return metrics[key]
    return None


def benchmark_comparison() -> None:
    formal = json.loads(FORMAL_JSON.read_text())
    benchmarks = [item["benchmark"] for item in formal["full_benchmarks"]]
    admission = {item["benchmark"]: item["metrics"].get("candidate_recall@100", 0.0) for item in formal["full_benchmarks"]}
    final = {
        item["benchmark"]: item["metrics"].get("final_recall@10", item["metrics"].get("recall_frac@10", 0.0))
        for item in formal["full_benchmarks"]
    }
    gap = {bench: admission[bench] - final[bench] for bench in benchmarks}
    body = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  ybar, width=0.98\linewidth, height=6.4cm, ymin=0, ymax=1.05,",
        r"  ylabel={Score}, symbolic x coords={"
        + ",".join(benchmarks)
        + r"}, xtick=data, x tick label style={rotate=20, anchor=east},",
        r"  legend style={at={(0.5,1.02)}, anchor=south, legend columns=3},",
        r"  bar width=5pt, enlarge x limits=0.15, grid=major, grid style={draw=gray!20}]",
    ]
    for series in [admission, final, gap]:
        body.append(r"\addplot coordinates {" + " ".join(f"({bench},{series[bench]:.4f})" for bench in benchmarks) + "};")
    body.append(r"\legend{Admission R@100,Final R@10,Admission-Final Gap}")
    body.extend([r"\end{axis}", r"\end{tikzpicture}"])
    write(FIG_DIR / "benchmark_comparison.pgf", "\n".join(body))


def efficiency_pareto() -> None:
    points = []
    with PARETO_CSV.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") != "available":
                continue
            elapsed = row.get("elapsed_seconds") or ""
            quality = row.get("recall_frac@10") or row.get("recall@10") or row.get("Recall@10")
            if not elapsed or not quality:
                continue
            try:
                points.append((float(elapsed), float(quality), row.get("benchmark", "run")))
            except ValueError:
                continue
    points = points[:80]
    if not points:
        write(
            FIG_DIR / "efficiency_pareto.pgf",
            r"""\begin{tikzpicture}
\node[draw=gray!60, rounded corners=2pt, align=center, text width=.82\linewidth] {Efficiency-quality Pareto figure is appendix-only: current artifacts do not expose matched quality metrics for every budget sweep without mixing non-comparable runs.};
\end{tikzpicture}""",
        )
        return
    body = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[width=0.96\linewidth,height=6cm,xlabel={Elapsed seconds},ylabel={Recall@10 / RecallFrac@10},grid=major,grid style={draw=gray!20}]",
        r"\addplot+[only marks, mark=*] coordinates {",
    ]
    body.extend(f"({x:.3f},{y:.4f})" for x, y, _ in points)
    body.extend([r"};", r"\end{axis}", r"\end{tikzpicture}"])
    write(FIG_DIR / "efficiency_pareto.pgf", "\n".join(body))


def failure_breakdown() -> None:
    formal = json.loads(FORMAL_JSON.read_text())
    coords = []
    for item in formal["full_benchmarks"]:
        metrics = item["metrics"]
        coords.append((item["benchmark"], metrics.get("candidate_recall@100", 0.0)))
    body = [
        r"\begin{tikzpicture}",
        r"\begin{axis}[ybar,width=0.9\linewidth,height=5.6cm,ymin=0,ymax=1.05,ylabel={Admission R@100},symbolic x coords={"
        + ",".join(name for name, _ in coords)
        + r"},xtick=data,x tick label style={rotate=20,anchor=east},grid=major,grid style={draw=gray!20}]",
        r"\addplot coordinates {" + " ".join(f"({name},{value:.4f})" for name, value in coords) + "};",
        r"\end{axis}",
        r"\end{tikzpicture}",
    ]
    write(FIG_DIR / "failure_breakdown.pgf", "\n".join(body))


def main() -> int:
    pipeline()
    benchmark_comparison()
    efficiency_pareto()
    failure_breakdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
