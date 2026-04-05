"""
Scorer Module
=============
Computes accuracy, authenticity, and up-to-dateness scores
from NLP evaluation results. Generates summary metrics.
"""

import json
import os
from typing import List, Dict
from datetime import datetime

import numpy as np
import pandas as pd


class Scorer:
    """Computes and manages evaluation scores for chatbot responses."""

    # Scoring thresholds for mapping NLP scores to Likert scale (0-5)
    LIKERT_THRESHOLDS = [
        (0.0, 0.2, 1),   # Very Poor
        (0.2, 0.4, 2),   # Poor
        (0.4, 0.6, 3),   # Average
        (0.6, 0.8, 4),   # Good
        (0.8, 1.0, 5),   # Excellent
    ]

    def __init__(self, config: Dict = None):
        self.config = config or {}
        eval_config = self.config.get("evaluation", {})
        self.scoring_config = eval_config.get("scoring", {})
        self.weights = self.scoring_config.get("weights", {
            "accuracy": 0.4,
            "authenticity": 0.3,
            "up_to_dateness": 0.3,
        })
        self.results_dir = self.config.get("paths", {}).get("results", "data/results/")
        os.makedirs(self.results_dir, exist_ok=True)

    def _nlp_to_likert(self, score: float) -> int:
        """Convert a 0-1 NLP score to 1-5 Likert scale."""
        score = max(0.0, min(1.0, score))
        for low, high, likert in self.LIKERT_THRESHOLDS:
            if low <= score < high:
                return likert
        return 5  # If score == 1.0

    def compute_accuracy_score(self, evaluation: Dict) -> Dict:
        """
        Compute accuracy score from NLP evaluation results.

        Accuracy measures factual alignment with ground truth, derived from:
        - Keyword similarity (TF-IDF)
        - Entity overlap (NER)
        - Semantic similarity (SBERT)
        """
        keyword_sim = evaluation.get("keyword_similarity", 0.0)
        entity_f1 = evaluation.get("entity_comparison", {}).get("entity_f1", 0.0)
        semantic_sim = evaluation.get("semantic_similarity", 0.0)
        keyword_overlap = evaluation.get("keyword_overlap", {}).get("overlap", 0.0)

        # Weighted combination for accuracy
        raw_score = (
            0.20 * keyword_sim
            + 0.15 * keyword_overlap
            + 0.25 * entity_f1
            + 0.40 * semantic_sim
        )

        return {
            "raw_score": round(raw_score, 4),
            "likert_score": self._nlp_to_likert(raw_score),
            "is_accurate": raw_score >= 0.5,
            "components": {
                "keyword_similarity": round(keyword_sim, 4),
                "keyword_overlap": round(keyword_overlap, 4),
                "entity_f1": round(entity_f1, 4),
                "semantic_similarity": round(semantic_sim, 4),
            },
        }

    def compute_authenticity_score(self, evaluation: Dict, response_text: str = "") -> Dict:
        """
        Compute authenticity score.

        Authenticity assesses whether the response appears to be generated
        from reliable knowledge rather than fabricated.

        Signals:
        - Consistent entity mentions (no contradictions)
        - Reasonable semantic similarity (not too low = gibberish, not repetitive)
        - Presence of specific factual entities
        - Response length and coherence
        """
        entity_data = evaluation.get("entity_comparison", {})
        semantic_sim = evaluation.get("semantic_similarity", 0.0)
        sentence_analysis = evaluation.get("sentence_analysis", {})
        avg_sent_sim = sentence_analysis.get("average_similarity", 0.0)
        coverage = sentence_analysis.get("coverage_above_07", 0.0)

        # Authenticity signals
        entity_count = entity_data.get("response_entities", 0)
        entity_specificity = min(entity_count / 5.0, 1.0)  # Normalize

        # Coherence: similarity should be moderate to high (not 0 = random, not too high = copied)
        coherence_score = min(semantic_sim * 1.2, 1.0)

        # Coverage of ground truth facts
        fact_coverage = coverage

        # Response substantiveness
        word_count = len(response_text.split()) if response_text else 0
        substantiveness = min(word_count / 50.0, 1.0)  # At least 50 words for full score

        raw_score = (
            0.30 * coherence_score
            + 0.25 * entity_specificity
            + 0.25 * fact_coverage
            + 0.20 * substantiveness
        )

        return {
            "raw_score": round(raw_score, 4),
            "likert_score": self._nlp_to_likert(raw_score),
            "is_authentic": raw_score >= 0.4,
            "components": {
                "coherence": round(coherence_score, 4),
                "entity_specificity": round(entity_specificity, 4),
                "fact_coverage": round(fact_coverage, 4),
                "substantiveness": round(substantiveness, 4),
            },
        }

    def compute_uptodate_score(
        self,
        evaluation: Dict,
        query: Dict,
        response_data: Dict = None,
    ) -> Dict:
        """
        Compute up-to-dateness score.

        Measures whether the response contains current information.

        For time-sensitive queries (Current Events, Recent Events):
        - Higher weight on temporal accuracy
        - Check if response references recent dates/events

        For stable factual queries:
        - Default to high score (information doesn't change)
        """
        query_type = query.get("type", "")
        domain = query.get("domain", "")
        gt_date_str = query.get("ground_truth_date", "")

        is_time_sensitive = (
            domain == "Current Events"
            or query_type == "Recent Events"
        )

        if is_time_sensitive:
            # For time-sensitive queries, use semantic similarity as proxy
            semantic_sim = evaluation.get("semantic_similarity", 0.0)
            entity_f1 = evaluation.get("entity_comparison", {}).get("entity_f1", 0.0)

            # Check for temporal entities in response
            entity_data = evaluation.get("entity_comparison", {})
            label_comparison = entity_data.get("label_comparison", {})
            date_entities = label_comparison.get("DATE", {})
            date_overlap = (
                date_entities.get("common_count", 0)
                / max(date_entities.get("ground_truth_count", 1), 1)
            )

            raw_score = (
                0.40 * semantic_sim
                + 0.30 * entity_f1
                + 0.30 * min(date_overlap, 1.0)
            )

            return {
                "raw_score": round(raw_score, 4),
                "likert_score": self._nlp_to_likert(raw_score),
                "is_uptodate": raw_score >= 0.4,
                "is_time_sensitive": True,
                "components": {
                    "semantic_match": round(semantic_sim, 4),
                    "entity_match": round(entity_f1, 4),
                    "temporal_match": round(date_overlap, 4),
                },
            }
        else:
            # For stable facts, accuracy ≈ up-to-dateness
            semantic_sim = evaluation.get("semantic_similarity", 0.0)
            raw_score = min(semantic_sim * 1.1, 1.0)  # Slight boost for stable facts

            return {
                "raw_score": round(raw_score, 4),
                "likert_score": self._nlp_to_likert(raw_score),
                "is_uptodate": raw_score >= 0.5,
                "is_time_sensitive": False,
                "components": {
                    "semantic_match": round(semantic_sim, 4),
                },
            }

    def score_all(
        self,
        evaluations: Dict,
        queries: List[Dict],
        responses: Dict = None,
    ) -> pd.DataFrame:
        """
        Score all evaluations and produce a comprehensive results DataFrame.

        Returns a DataFrame with one row per (query, chatbot) pair.
        """
        query_lookup = {q["id"]: q for q in queries}
        records = []

        for query_id, chatbot_evaluations in evaluations.items():
            query = query_lookup.get(query_id, {})

            for chatbot_id, evaluation in chatbot_evaluations.items():
                response_text = ""
                if responses and query_id in responses:
                    resp_data = responses[query_id].get(chatbot_id, {})
                    response_text = resp_data.get("response", "")

                accuracy = self.compute_accuracy_score(evaluation)
                authenticity = self.compute_authenticity_score(evaluation, response_text)
                uptodate = self.compute_uptodate_score(evaluation, query)

                # Weighted overall score
                overall = (
                    self.weights["accuracy"] * accuracy["raw_score"]
                    + self.weights["authenticity"] * authenticity["raw_score"]
                    + self.weights["up_to_dateness"] * uptodate["raw_score"]
                )

                records.append({
                    "query_id": query_id,
                    "chatbot_id": chatbot_id,
                    "domain": query.get("domain", ""),
                    "query_type": query.get("type", ""),
                    "difficulty": query.get("difficulty", ""),

                    # Raw scores (0-1)
                    "accuracy_raw": accuracy["raw_score"],
                    "authenticity_raw": authenticity["raw_score"],
                    "uptodate_raw": uptodate["raw_score"],
                    "overall_raw": round(overall, 4),

                    # Likert scores (1-5)
                    "accuracy_likert": accuracy["likert_score"],
                    "authenticity_likert": authenticity["likert_score"],
                    "uptodate_likert": uptodate["likert_score"],

                    # Binary flags
                    "is_accurate": accuracy["is_accurate"],
                    "is_authentic": authenticity["is_authentic"],
                    "is_uptodate": uptodate["is_uptodate"],

                    # NLP component scores
                    "keyword_similarity": evaluation.get("keyword_similarity", 0.0),
                    "entity_f1": evaluation.get("entity_comparison", {}).get("entity_f1", 0.0),
                    "semantic_similarity": evaluation.get("semantic_similarity", 0.0),
                    "nlp_overall": evaluation.get("overall_nlp_score", 0.0),
                })

        df = pd.DataFrame(records)

        # Save to CSV
        output_path = os.path.join(self.results_dir, "evaluation_scores.csv")
        df.to_csv(output_path, index=False)
        print(f"\n[Scorer] Evaluation scores saved to {output_path}")
        print(f"  → Total entries: {len(df)}")

        return df

    def compute_summary_statistics(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Compute summary statistics per chatbot."""
        metrics = [
            "accuracy_raw", "authenticity_raw", "uptodate_raw", "overall_raw",
            "keyword_similarity", "entity_f1", "semantic_similarity",
        ]

        summary = scores_df.groupby("chatbot_id")[metrics].agg(["mean", "std", "median"]).round(4)
        summary.columns = ["_".join(col) for col in summary.columns]

        # Add accuracy/up-to-dateness rates
        rates = scores_df.groupby("chatbot_id").agg(
            accuracy_rate=("is_accurate", "mean"),
            authenticity_rate=("is_authentic", "mean"),
            uptodate_rate=("is_uptodate", "mean"),
            total_queries=("query_id", "count"),
        ).round(4)

        summary = summary.join(rates)

        # Save
        output_path = os.path.join(self.results_dir, "summary_statistics.csv")
        summary.to_csv(output_path)
        print(f"[Scorer] Summary statistics saved to {output_path}")

        return summary

    def compute_domain_breakdown(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-domain performance breakdown."""
        breakdown = scores_df.groupby(["chatbot_id", "domain"]).agg(
            accuracy_mean=("accuracy_raw", "mean"),
            authenticity_mean=("authenticity_raw", "mean"),
            uptodate_mean=("uptodate_raw", "mean"),
            overall_mean=("overall_raw", "mean"),
            count=("query_id", "count"),
        ).round(4)

        output_path = os.path.join(self.results_dir, "domain_breakdown.csv")
        breakdown.to_csv(output_path)
        print(f"[Scorer] Domain breakdown saved to {output_path}")

        return breakdown
