"""Benchmark utilities for latency and citation coverage metrics."""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

# Allow running from project root with: python src/benchmark.py
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import create_qa_agent, query_agent
from src.data_pipeline import pipeline
from src.retriever import get_retriever


def load_questions(path: Path) -> list[str]:
    """Load non-empty, non-comment lines as benchmark questions."""
    questions: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        questions.append(item)
    return questions


def percentile(values: list[float], p: float) -> float:
    """Compute percentile using linear interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return ordered[lower]

    lower_val = ordered[lower]
    upper_val = ordered[upper]
    frac = idx - lower
    return lower_val + (upper_val - lower_val) * frac


def run_benchmark(
    questions_file: Path,
    ingest_query: str,
    max_articles: int,
    skip_ingestion: bool,
    output_file: Path,
) -> dict:
    """Run benchmark and return summary dictionary."""
    questions = load_questions(questions_file)
    if not questions:
        raise ValueError(f"No questions found in {questions_file}")

    retriever = get_retriever()

    if not skip_ingestion:
        print(f"\nLoading benchmark corpus for query: {ingest_query}")
        documents = pipeline(ingest_query, max_articles=max_articles)
        if documents:
            retriever.vector_store.add_documents(documents)
            print(f"Loaded {len(documents)} chunks for benchmarking.\n")
        else:
            print("No documents loaded. Benchmark will run on current vector store.\n")

    print("Creating QA agent...")
    agent = create_qa_agent(memory_enabled=False)
    print(f"Running {len(questions)} benchmark questions...\n")

    latencies: list[float] = []
    citation_hits = 0
    results: list[dict] = []

    for i, question in enumerate(questions, start=1):
        start = time.perf_counter()
        answer = query_agent(agent, question)
        elapsed = time.perf_counter() - start

        has_pmid = bool(re.search(r"\[PMID:\d+\]", answer))
        citation_hits += 1 if has_pmid else 0
        latencies.append(elapsed)

        results.append(
            {
                "index": i,
                "question": question,
                "latency_seconds": round(elapsed, 3),
                "has_pmid_citation": has_pmid,
                "answer": answer,
            }
        )
        print(f"[{i}/{len(questions)}] {elapsed:.2f}s | cited={has_pmid}")

    avg_latency = sum(latencies) / len(latencies)
    p95_latency = percentile(latencies, 0.95)
    citation_coverage = (citation_hits / len(questions)) * 100.0

    summary = {
        "num_questions": len(questions),
        "avg_latency_seconds": round(avg_latency, 3),
        "p95_latency_seconds": round(p95_latency, 3),
        "citation_coverage_percent": round(citation_coverage, 1),
        "answers_with_pmid": citation_hits,
        "total_answers": len(questions),
    }

    payload = {
        "summary": summary,
        "questions_file": str(questions_file),
        "ingest_query": ingest_query,
        "max_articles": max_articles,
        "results": results,
    }
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nBenchmark summary")
    print("-" * 50)
    print(f"Average latency: {summary['avg_latency_seconds']}s")
    print(f"P95 latency: {summary['p95_latency_seconds']}s")
    print(
        "Citation coverage: "
        f"{summary['citation_coverage_percent']}% "
        f"({summary['answers_with_pmid']}/{summary['total_answers']})"
    )
    print(f"Saved results to: {output_file}")

    return payload


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compute latency and citation coverage metrics for the RAG system."
    )
    parser.add_argument(
        "--questions-file",
        default="benchmark_questions.txt",
        help="Path to newline-separated benchmark questions.",
    )
    parser.add_argument(
        "--ingest-query",
        default="fatty liver",
        help="PubMed query used to load benchmark corpus before evaluation.",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=50,
        help="Maximum number of PubMed articles to ingest for benchmarking.",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip corpus ingestion and evaluate against current vector store.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Path to write benchmark JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    run_benchmark(
        questions_file=Path(args.questions_file),
        ingest_query=args.ingest_query,
        max_articles=args.max_articles,
        skip_ingestion=args.skip_ingestion,
        output_file=Path(args.output),
    )


if __name__ == "__main__":
    main()
