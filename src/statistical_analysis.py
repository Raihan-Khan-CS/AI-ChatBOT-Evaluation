"""
Statistical Analysis Module
============================
Performs statistical tests to determine significance of performance
differences between chatbots.
"""

import json
import os
from typing import Dict, List, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


class StatisticalAnalyzer:
    """Performs statistical analysis on chatbot evaluation scores."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        stat_config = self.config.get("statistics", {})
        self.alpha = stat_config.get("significance_level", 0.05)
        self.correction = stat_config.get("correction_method", "bonferroni")
        self.results_dir = self.config.get("paths", {}).get("results", "data/results/")
        os.makedirs(self.results_dir, exist_ok=True)

    def run_anova(self, scores_df: pd.DataFrame, metric: str = "overall_raw") -> Dict:
        """
        Run one-way ANOVA to test if there are significant differences
        among chatbot performance means.
        """
        groups = []
        chatbot_ids = sorted(scores_df["chatbot_id"].unique())

        for cb_id in chatbot_ids:
            group_scores = scores_df[scores_df["chatbot_id"] == cb_id][metric].dropna().values
            if len(group_scores) > 0:
                groups.append(group_scores)

        if len(groups) < 2:
            return {
                "test": "One-way ANOVA",
                "metric": metric,
                "error": "Need at least 2 groups",
            }

        f_stat, p_value = stats.f_oneway(*groups)

        return {
            "test": "One-way ANOVA",
            "metric": metric,
            "f_statistic": round(float(f_stat), 4),
            "p_value": round(float(p_value), 6),
            "significant": p_value < self.alpha,
            "alpha": self.alpha,
            "groups": chatbot_ids,
            "group_sizes": [len(g) for g in groups],
            "group_means": {
                cb: round(float(np.mean(g)), 4)
                for cb, g in zip(chatbot_ids, groups)
            },
        }

    def run_pairwise_ttests(
        self,
        scores_df: pd.DataFrame,
        metric: str = "overall_raw",
    ) -> List[Dict]:
        """
        Run pairwise independent t-tests between all chatbot pairs
        with Bonferroni correction.
        """
        chatbot_ids = sorted(scores_df["chatbot_id"].unique())
        pairs = list(combinations(chatbot_ids, 2))
        n_comparisons = len(pairs)
        corrected_alpha = self.alpha / n_comparisons if n_comparisons > 0 else self.alpha

        results = []
        for cb1, cb2 in pairs:
            scores1 = scores_df[scores_df["chatbot_id"] == cb1][metric].dropna().values
            scores2 = scores_df[scores_df["chatbot_id"] == cb2][metric].dropna().values

            if len(scores1) < 2 or len(scores2) < 2:
                results.append({
                    "pair": f"{cb1} vs {cb2}",
                    "error": "Insufficient data",
                })
                continue

            t_stat, p_value = stats.ttest_ind(scores1, scores2, equal_var=False)

            results.append({
                "pair": f"{cb1} vs {cb2}",
                "chatbot_1": cb1,
                "chatbot_2": cb2,
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_value), 6),
                "p_value_corrected": round(float(p_value * n_comparisons), 6),
                "significant_raw": p_value < self.alpha,
                "significant_corrected": (p_value * n_comparisons) < self.alpha,
                "mean_1": round(float(np.mean(scores1)), 4),
                "mean_2": round(float(np.mean(scores2)), 4),
                "mean_diff": round(float(np.mean(scores1) - np.mean(scores2)), 4),
                "effect_size_cohens_d": round(
                    float(self._cohens_d(scores1, scores2)), 4
                ),
            })

        return results

    def run_kruskal_wallis(self, scores_df: pd.DataFrame, metric: str = "overall_raw") -> Dict:
        """
        Run Kruskal-Wallis H-test (non-parametric alternative to ANOVA).
        Useful when assumptions of normality aren't met.
        """
        groups = []
        chatbot_ids = sorted(scores_df["chatbot_id"].unique())

        for cb_id in chatbot_ids:
            group_scores = scores_df[scores_df["chatbot_id"] == cb_id][metric].dropna().values
            if len(group_scores) > 0:
                groups.append(group_scores)

        if len(groups) < 2:
            return {"test": "Kruskal-Wallis", "error": "Need at least 2 groups"}

        h_stat, p_value = stats.kruskal(*groups)

        return {
            "test": "Kruskal-Wallis H-test",
            "metric": metric,
            "h_statistic": round(float(h_stat), 4),
            "p_value": round(float(p_value), 6),
            "significant": p_value < self.alpha,
            "alpha": self.alpha,
        }

    def compute_correlation_matrix(self, scores_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute correlation matrix between evaluation dimensions.
        """
        metrics = ["accuracy_raw", "authenticity_raw", "uptodate_raw",
                    "keyword_similarity", "entity_f1", "semantic_similarity"]

        available_metrics = [m for m in metrics if m in scores_df.columns]
        corr_matrix = scores_df[available_metrics].corr(method="pearson").round(4)

        output_path = os.path.join(self.results_dir, "correlation_matrix.csv")
        corr_matrix.to_csv(output_path)

        return corr_matrix

    def normality_test(self, scores_df: pd.DataFrame, metric: str = "overall_raw") -> Dict:
        """Run Shapiro-Wilk normality test per chatbot group."""
        results = {}
        for cb_id in sorted(scores_df["chatbot_id"].unique()):
            scores = scores_df[scores_df["chatbot_id"] == cb_id][metric].dropna().values
            if len(scores) >= 3:
                stat, p_value = stats.shapiro(scores)
                results[cb_id] = {
                    "w_statistic": round(float(stat), 4),
                    "p_value": round(float(p_value), 6),
                    "is_normal": p_value > self.alpha,
                }
            else:
                results[cb_id] = {"error": "Too few samples for normality test"}

        return results

    def run_full_analysis(self, scores_df: pd.DataFrame) -> Dict:
        """
        Run complete statistical analysis suite.
        """
        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS")
        print("=" * 60)

        metrics = ["accuracy_raw", "authenticity_raw", "uptodate_raw", "overall_raw"]
        full_results = {}

        for metric in metrics:
            print(f"\n--- Analyzing: {metric} ---")

            # Normality test
            normality = self.normality_test(scores_df, metric)

            # ANOVA
            anova = self.run_anova(scores_df, metric)
            print(f"  ANOVA: F={anova.get('f_statistic', 'N/A')}, "
                  f"p={anova.get('p_value', 'N/A')}, "
                  f"significant={anova.get('significant', 'N/A')}")

            # Kruskal-Wallis (non-parametric)
            kruskal = self.run_kruskal_wallis(scores_df, metric)

            # Pairwise t-tests
            pairwise = self.run_pairwise_ttests(scores_df, metric)
            sig_pairs = [p["pair"] for p in pairwise
                         if isinstance(p.get("significant_corrected"), bool)
                         and p["significant_corrected"]]
            print(f"  Significant pairwise differences: {len(sig_pairs)}")

            full_results[metric] = {
                "normality": normality,
                "anova": anova,
                "kruskal_wallis": kruskal,
                "pairwise_ttests": pairwise,
                "significant_pairs": sig_pairs,
            }

        # Correlation matrix
        corr_matrix = self.compute_correlation_matrix(scores_df)
        full_results["correlation_matrix"] = corr_matrix.to_dict()

        # Save full results
        output_path = os.path.join(self.results_dir, "statistical_tests.json")
        # Convert numpy types for JSON serialization
        serializable = json.loads(json.dumps(full_results, default=str))
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n[StatisticalAnalyzer] Results saved to {output_path}")

        return full_results

    # ----------------------------------------------------------------
    # Utility Methods
    # ----------------------------------------------------------------

    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def print_summary(self, results: Dict):
        """Print a human-readable summary of statistical analysis."""
        print("\n" + "=" * 60)
        print("STATISTICAL ANALYSIS SUMMARY")
        print("=" * 60)

        for metric, data in results.items():
            if metric == "correlation_matrix":
                continue
            print(f"\n{'─' * 40}")
            print(f"Metric: {metric}")
            print(f"{'─' * 40}")

            anova = data.get("anova", {})
            if "f_statistic" in anova:
                sig = "YES ✓" if anova["significant"] else "NO ✗"
                print(f"  ANOVA: F={anova['f_statistic']}, p={anova['p_value']} → {sig}")
                if anova.get("group_means"):
                    print(f"  Means: {anova['group_means']}")

            sig_pairs = data.get("significant_pairs", [])
            if sig_pairs:
                print(f"  Significant pairs (Bonferroni): {', '.join(sig_pairs)}")
            else:
                print("  No significant pairwise differences found")
