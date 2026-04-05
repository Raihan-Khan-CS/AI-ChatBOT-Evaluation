"""
Visualizer Module
=================
Generates charts, plots, and visual reports for chatbot
evaluation results.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    """Generates visualizations for chatbot evaluation results."""

    # Default chatbot colors
    DEFAULT_COLORS = {
        "chatgpt": "#10A37F",
        "gemini": "#4285F4",
        "claude": "#D97757",
        "grok": "#1DA1F2",
        "deepseek": "#4A90D9",
    }

    CHATBOT_LABELS = {
        "chatgpt": "ChatGPT",
        "gemini": "Gemini",
        "claude": "Claude",
        "grok": "Grok",
        "deepseek": "DeepSeek",
    }

    def __init__(self, config: Dict = None):
        self.config = config or {}
        vis_config = self.config.get("visualization", {})
        self.figsize = tuple(vis_config.get("figsize", [12, 8]))
        self.dpi = vis_config.get("dpi", 150)
        self.colors = vis_config.get("colors", self.DEFAULT_COLORS)
        self.figures_dir = self.config.get("paths", {}).get("figures", "data/results/figures/")
        os.makedirs(self.figures_dir, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "figure.dpi": self.dpi,
        })

    def _get_color(self, chatbot_id: str) -> str:
        return self.colors.get(chatbot_id, self.DEFAULT_COLORS.get(chatbot_id, "#888888"))

    def _get_label(self, chatbot_id: str) -> str:
        return self.CHATBOT_LABELS.get(chatbot_id, chatbot_id.title())

    def _save_fig(self, fig: plt.Figure, filename: str):
        path = os.path.join(self.figures_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  → Saved: {path}")

    # ================================================================
    # 1. Overall Comparison Bar Chart
    # ================================================================

    def plot_overall_comparison(self, scores_df: pd.DataFrame):
        """Bar chart comparing chatbot performance across all dimensions."""
        chatbots = sorted(scores_df["chatbot_id"].unique())
        metrics = {
            "accuracy_raw": "Accuracy",
            "authenticity_raw": "Authenticity",
            "uptodate_raw": "Up-to-dateness",
        }

        means = scores_df.groupby("chatbot_id")[list(metrics.keys())].mean()

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(chatbots))
        width = 0.25

        for i, (metric_col, metric_label) in enumerate(metrics.items()):
            values = [means.loc[cb, metric_col] if cb in means.index else 0 for cb in chatbots]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=metric_label, alpha=0.85)
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9
                )

        ax.set_xlabel("Chatbot")
        ax.set_ylabel("Score (0-1)")
        ax.set_title("Comparison of Chatbot Performance Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels([self._get_label(cb) for cb in chatbots])
        ax.set_ylim(0, 1.15)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

        self._save_fig(fig, "01_overall_comparison.png")

    # ================================================================
    # 2. Radar Chart
    # ================================================================

    def plot_radar_chart(self, scores_df: pd.DataFrame):
        """Radar/spider chart for multi-dimensional comparison."""
        chatbots = sorted(scores_df["chatbot_id"].unique())
        metrics = ["accuracy_raw", "authenticity_raw", "uptodate_raw",
                    "keyword_similarity", "entity_f1", "semantic_similarity"]
        labels = ["Accuracy", "Authenticity", "Up-to-dateness",
                   "Keyword\nSimilarity", "Entity F1", "Semantic\nSimilarity"]

        means = scores_df.groupby("chatbot_id")[metrics].mean()

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        for cb in chatbots:
            if cb not in means.index:
                continue
            values = means.loc[cb].tolist()
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=self._get_label(cb),
                    color=self._get_color(cb))
            ax.fill(angles, values, alpha=0.1, color=self._get_color(cb))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=10)
        ax.set_ylim(0, 1)
        ax.set_title("Multi-Dimensional Chatbot Performance Comparison", size=14, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        self._save_fig(fig, "02_radar_chart.png")

    # ================================================================
    # 3. Box Plot
    # ================================================================

    def plot_boxplots(self, scores_df: pd.DataFrame):
        """Box plots showing score distributions per chatbot."""
        metrics = {
            "accuracy_raw": "Accuracy Score",
            "authenticity_raw": "Authenticity Score",
            "uptodate_raw": "Up-to-dateness Score",
            "overall_raw": "Overall Score",
        }

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        chatbots = sorted(scores_df["chatbot_id"].unique())
        palette = {cb: self._get_color(cb) for cb in chatbots}

        for idx, (metric, title) in enumerate(metrics.items()):
            plot_df = scores_df[["chatbot_id", metric]].copy()
            plot_df["chatbot_id"] = plot_df["chatbot_id"].map(
                lambda x: self._get_label(x)
            )
            renamed_palette = {self._get_label(k): v for k, v in palette.items()}

            sns.boxplot(
                data=plot_df, x="chatbot_id", y=metric,
                palette=renamed_palette, ax=axes[idx],
                order=[self._get_label(cb) for cb in chatbots]
            )
            axes[idx].set_title(title)
            axes[idx].set_xlabel("")
            axes[idx].set_ylabel("Score")
            axes[idx].set_ylim(-0.05, 1.05)

        fig.suptitle("Score Distributions by Chatbot", fontsize=16, y=1.02)
        plt.tight_layout()
        self._save_fig(fig, "03_boxplots.png")

    # ================================================================
    # 4. Domain Breakdown Heatmap
    # ================================================================

    def plot_domain_heatmap(self, scores_df: pd.DataFrame):
        """Heatmap showing performance across domains for each chatbot."""
        pivot = scores_df.pivot_table(
            values="overall_raw",
            index="chatbot_id",
            columns="domain",
            aggfunc="mean"
        )

        # Rename index for display
        pivot.index = [self._get_label(cb) for cb in pivot.index]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, vmin=0, vmax=1,
            cbar_kws={"label": "Overall Score"}
        )
        ax.set_title("Chatbot Performance by Domain")
        ax.set_ylabel("Chatbot")
        ax.set_xlabel("Domain")

        self._save_fig(fig, "04_domain_heatmap.png")

    # ================================================================
    # 5. Accuracy Rate Bar Chart
    # ================================================================

    def plot_accuracy_rates(self, scores_df: pd.DataFrame):
        """Bar chart showing accuracy and up-to-dateness rates."""
        chatbots = sorted(scores_df["chatbot_id"].unique())

        rates = scores_df.groupby("chatbot_id").agg(
            accuracy_rate=("is_accurate", "mean"),
            uptodate_rate=("is_uptodate", "mean"),
        )

        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(chatbots))
        width = 0.35

        acc_vals = [rates.loc[cb, "accuracy_rate"] * 100 if cb in rates.index else 0 for cb in chatbots]
        upd_vals = [rates.loc[cb, "uptodate_rate"] * 100 if cb in rates.index else 0 for cb in chatbots]

        bars1 = ax.bar(x - width / 2, acc_vals, width, label="Accuracy Rate (%)",
                       color="#FFD700", alpha=0.85, edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, upd_vals, width, label="Up-to-dateness Rate (%)",
                       color="#FFA500", alpha=0.85, edgecolor="black", linewidth=0.5)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                        f"{height:.1f}%", ha="center", va="bottom", fontsize=10)

        ax.set_xlabel("Chatbot")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Comparison of Chatbot Performance Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels([self._get_label(cb) for cb in chatbots])
        ax.set_ylim(0, 110)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        self._save_fig(fig, "05_accuracy_rates.png")

    # ================================================================
    # 6. Statistical Significance Plot
    # ================================================================

    def plot_ttest_results(self, stat_results: Dict, metric: str = "overall_raw"):
        """Plot t-test results showing p-values vs significance threshold."""
        pairwise = stat_results.get(metric, {}).get("pairwise_ttests", [])
        if not pairwise:
            return

        valid = [p for p in pairwise if "t_statistic" in p]
        if not valid:
            return

        fig, ax = plt.subplots(figsize=(10, 7))

        t_vals = [p["t_statistic"] for p in valid]
        p_vals = [p["p_value"] for p in valid]
        labels = [p["pair"] for p in valid]

        colors = ["#FF4444" if p["significant_raw"] else "#4444FF" for p in valid]

        ax.scatter(t_vals, p_vals, c=colors, s=120, marker="x", linewidths=2, zorder=5)
        ax.axhline(y=0.05, color="blue", linestyle="--", linewidth=1.5,
                   label="Significance Threshold (p=0.05)")

        for t, p, label in zip(t_vals, p_vals, labels):
            ax.annotate(label, (t, p), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)

        ax.set_xlabel("T-Values")
        ax.set_ylabel("P-Values")
        ax.set_title(f"Statistical Significance Test (T-Test) — {metric}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        self._save_fig(fig, "06_ttest_significance.png")

    # ================================================================
    # 7. Correlation Heatmap
    # ================================================================

    def plot_correlation_heatmap(self, scores_df: pd.DataFrame):
        """Plot correlation matrix between evaluation metrics."""
        metrics = ["accuracy_raw", "authenticity_raw", "uptodate_raw",
                   "keyword_similarity", "entity_f1", "semantic_similarity"]
        available = [m for m in metrics if m in scores_df.columns]

        corr = scores_df[available].corr()

        labels = {
            "accuracy_raw": "Accuracy",
            "authenticity_raw": "Authenticity",
            "uptodate_raw": "Up-to-dateness",
            "keyword_similarity": "Keyword Sim",
            "entity_f1": "Entity F1",
            "semantic_similarity": "Semantic Sim",
        }
        corr.index = [labels.get(m, m) for m in corr.index]
        corr.columns = [labels.get(m, m) for m in corr.columns]

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax,
            cbar_kws={"label": "Pearson Correlation"}
        )
        ax.set_title("Correlation Between Evaluation Metrics")

        self._save_fig(fig, "07_correlation_heatmap.png")

    # ================================================================
    # 8. Query Type Analysis
    # ================================================================

    def plot_query_type_analysis(self, scores_df: pd.DataFrame):
        """Grouped bar chart showing performance by query type."""
        chatbots = sorted(scores_df["chatbot_id"].unique())
        query_types = sorted(scores_df["query_type"].unique())

        pivot = scores_df.pivot_table(
            values="overall_raw",
            index="query_type",
            columns="chatbot_id",
            aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=(14, 7))
        pivot.rename(columns={cb: self._get_label(cb) for cb in chatbots}).plot(
            kind="bar", ax=ax,
            color=[self._get_color(cb) for cb in chatbots],
            alpha=0.85, edgecolor="black", linewidth=0.5
        )

        ax.set_xlabel("Query Type")
        ax.set_ylabel("Overall Score")
        ax.set_title("Chatbot Performance by Query Type")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_ylim(0, 1.1)
        ax.legend(title="Chatbot")
        ax.grid(axis="y", alpha=0.3)

        self._save_fig(fig, "08_query_type_analysis.png")

    # ================================================================
    # Generate All Figures
    # ================================================================

    def generate_all(self, scores_df: pd.DataFrame, stat_results: Dict = None):
        """Generate all visualization figures."""
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        print("\n1. Overall Comparison...")
        self.plot_overall_comparison(scores_df)

        print("2. Radar Chart...")
        self.plot_radar_chart(scores_df)

        print("3. Box Plots...")
        self.plot_boxplots(scores_df)

        print("4. Domain Heatmap...")
        self.plot_domain_heatmap(scores_df)

        print("5. Accuracy Rates...")
        self.plot_accuracy_rates(scores_df)

        if stat_results:
            print("6. T-Test Significance...")
            self.plot_ttest_results(stat_results)

        print("7. Correlation Heatmap...")
        self.plot_correlation_heatmap(scores_df)

        print("8. Query Type Analysis...")
        self.plot_query_type_analysis(scores_df)

        print(f"\nAll figures saved to: {self.figures_dir}")
