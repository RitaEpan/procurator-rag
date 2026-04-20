import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import NUM_SIMILAR_EXAMPLES
from src.data_loader import load_dataset
from src.generator import AnswerGenerator
from src.search_engine import SearchEngine
from src.vectorizer import Vectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare response generation quality without RAG and with RAG."
    )
    parser.add_argument(
        "--input",
        default="data/data.csv",
        help="Path to a CSV file with complaint and response columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for experiment outputs.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Share of records used as the test set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducible splitting.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=NUM_SIMILAR_EXAMPLES,
        help="Number of similar examples to use in RAG.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of test examples for a quick run.",
    )
    return parser.parse_args()


def validate_dataset(df: pd.DataFrame) -> None:
    required_columns = {"complaint", "response"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required dataset columns: {missing_str}")


def split_dataset(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be in the (0, 1) range")

    if len(df) < 2:
        raise ValueError("At least 2 dataset records are required for comparison")

    rng = np.random.default_rng(random_state)
    shuffled_indices = rng.permutation(len(df))
    test_count = max(1, int(round(len(df) * test_size)))
    test_indices = shuffled_indices[:test_count]
    train_indices = shuffled_indices[test_count:]

    if len(train_indices) == 0:
        raise ValueError("The training set is empty after splitting")

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    return train_df, test_df


def build_baseline_prompt(complaint: str) -> str:
    return f"""
You are an intelligent assistant for a prosecutor's office employee.
Your task is to draft an official response to a citizen complaint.

STYLE: Strictly formal and businesslike. Do not use emotional language or first-person wording.
Write the final response in Russian.
Use standard prosecutor-office wording for reviewed complaints, established facts, explanations, and appeal rights.

TASK:
Write an official response to the following complaint:
"{complaint}"

Your response:
    """.strip()


def rouge_l_f1(reference: str, candidate: str) -> float:
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if not ref_tokens or not cand_tokens:
        return 0.0

    lcs = lcs_length(ref_tokens, cand_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def tokenize(text: str) -> list[str]:
    return str(text).lower().split()


def lcs_length(first: Iterable[str], second: Iterable[str]) -> int:
    first = list(first)
    second = list(second)
    previous = [0] * (len(second) + 1)

    for token_first in first:
        current = [0]
        for index, token_second in enumerate(second, start=1):
            if token_first == token_second:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current

    return previous[-1]


def evaluate_pair(reference: str, baseline: str, rag: str) -> dict:
    baseline_rouge = rouge_l_f1(reference, baseline)
    rag_rouge = rouge_l_f1(reference, rag)
    return {
        "baseline_rouge_l_f1": baseline_rouge,
        "rag_rouge_l_f1": rag_rouge,
        "rag_minus_baseline": rag_rouge - baseline_rouge,
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.input)
    validate_dataset(df)
    train_df, test_df = split_dataset(df, test_size=args.test_size, random_state=args.random_state)

    if args.limit is not None:
        test_df = test_df.head(args.limit).reset_index(drop=True)

    retrieval_column = "complaint_anon" if "complaint_anon" in train_df.columns else "complaint"

    print(f"Total records: {len(df)}")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Retrieval column: {retrieval_column}")

    vectorizer = Vectorizer()
    train_embeddings = vectorizer.encode_texts(train_df[retrieval_column].fillna("").tolist())
    search_engine = SearchEngine(train_df, train_embeddings)
    generator = AnswerGenerator()

    rows = []

    for index, row in test_df.iterrows():
        complaint = str(row["complaint"])
        reference_response = str(row["response"])

        print(f"[{index + 1}/{len(test_df)}] Generating responses for a test complaint...")

        baseline_prompt = build_baseline_prompt(complaint)
        baseline_response = generator.generate(baseline_prompt)

        query_vector = vectorizer.encode_single(complaint)
        rag_examples = search_engine.find_similar(query_vector, top_k=args.top_k)
        rag_prompt = generator.create_prompt(complaint, rag_examples)
        rag_response = generator.generate(rag_prompt)

        metrics = evaluate_pair(reference_response, baseline_response, rag_response)

        rows.append(
            {
                "complaint": complaint,
                "reference_response": reference_response,
                "baseline_response": baseline_response,
                "rag_response": rag_response,
                "retrieved_examples": json.dumps(rag_examples, ensure_ascii=False),
                **metrics,
                "baseline_manual_score": "",
                "rag_manual_score": "",
                "baseline_style_score": "",
                "rag_style_score": "",
                "baseline_completeness_score": "",
                "rag_completeness_score": "",
                "notes": "",
            }
        )

    results_df = pd.DataFrame(rows)
    results_path = output_dir / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

    summary = {
        "dataset_size": int(len(df)),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "top_k": int(args.top_k),
        "random_state": int(args.random_state),
        "baseline_rouge_l_f1_mean": float(results_df["baseline_rouge_l_f1"].mean()),
        "rag_rouge_l_f1_mean": float(results_df["rag_rouge_l_f1"].mean()),
        "rag_minus_baseline_mean": float(results_df["rag_minus_baseline"].mean()),
    }

    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nExperiment completed.")
    print(f"Results CSV: {results_path}")
    print(f"Summary JSON: {summary_path}")
    print(
        "Average ROUGE-L F1:"
        f" baseline={summary['baseline_rouge_l_f1_mean']:.4f},"
        f" rag={summary['rag_rouge_l_f1_mean']:.4f},"
        f" delta={summary['rag_minus_baseline_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
