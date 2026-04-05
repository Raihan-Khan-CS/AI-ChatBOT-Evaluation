#!/usr/bin/env python3
"""
AI Chatbot Evaluation Framework — Main Entry Point
====================================================
CSE 533 - Machine Learning Project

Usage:
    python main.py --mode full          # Run complete pipeline
    python main.py --mode collect       # Collect responses only
    python main.py --mode evaluate      # Evaluate collected responses
    python main.py --mode analyze       # Statistical analysis & visualization
    python main.py --mode demo          # Run with sample/simulated data
    python main.py --mode template      # Generate manual collection template
"""

import argparse
import json
import os
import sys
import yaml
from datetime import datetime

from src.query_manager import QueryManager
from src.chatbot_interface import ChatbotInterface
from src.ground_truth_collector import GroundTruthCollector
from src.nlp_evaluator import NLPEvaluator
from src.scorer import Scorer
from src.statistical_analysis import StatisticalAnalyzer
from src.visualizer import Visualizer


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    # Try local config first
    local_path = config_path.replace(".yaml", "_local.yaml")
    if os.path.exists(local_path):
        config_path = local_path
        print(f"[Config] Using local config: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def print_banner():
    """Print project banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║       AI Chatbot Evaluation Framework v1.0.0                ║
║                                                              ║
║  Investigating Accuracy, Authenticity, and Up-to-dateness    ║
║  in Conversational AI Systems                                ║
║                                                              ║
║  CSE 533 — Machine Learning                                  ║
║  MD Raihan Khan & Zannatul Islam Proma                       ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_collect(config: dict):
    """Stage 1: Collect chatbot responses."""
    print("\n" + "=" * 60)
    print("STAGE 1: COLLECTING CHATBOT RESPONSES")
    print("=" * 60)

    qm = QueryManager(config["paths"]["queries"])
    queries = qm.get_all_queries()
    print(f"\nDataset Statistics: {qm.get_statistics()}")

    ci = ChatbotInterface(config)

    # Check for API keys
    available = []
    for cb in config["chatbots"]:
        key = os.environ.get(cb["api_key_env"], "")
        status = "✓ Found" if key else "✗ Missing"
        print(f"  {cb['name']:12s}: {status}")
        if key:
            available.append(cb["id"])

    if not available:
        print("\n⚠ No API keys found. Generating manual collection template...")
        ci.generate_manual_collection_template(queries)
        print("\nTo proceed:")
        print("  1. Manually query each chatbot and fill in the template")
        print("  2. Run: python main.py --mode evaluate")
        return

    # Collect responses
    responses = ci.collect_all_responses(queries)
    print(f"\nCollected responses for {len(responses)} queries")

    # Also collect ground truth
    gtc = GroundTruthCollector(config)
    gt_data = gtc.extract_from_queries(queries)
    print(f"Extracted {len(gt_data)} ground truth entries")


def run_evaluate(config: dict):
    """Stage 2: Evaluate responses using NLP techniques."""
    print("\n" + "=" * 60)
    print("STAGE 2: NLP EVALUATION")
    print("=" * 60)

    qm = QueryManager(config["paths"]["queries"])
    queries = qm.get_all_queries()

    # Load responses
    responses_path = os.path.join(config["paths"]["responses"], "chatbot_responses.json")
    manual_path = os.path.join(config["paths"]["responses"], "manual_collection_template.json")

    ci = ChatbotInterface(config)

    if os.path.exists(responses_path):
        responses = ci.load_responses(responses_path)
        print(f"Loaded API responses from {responses_path}")
    elif os.path.exists(manual_path):
        responses = ci.load_manual_responses(manual_path)
        print(f"Loaded manual responses from {manual_path}")
    else:
        print("ERROR: No response files found!")
        print(f"  Expected: {responses_path}")
        print(f"       or: {manual_path}")
        sys.exit(1)

    # Load/create ground truth
    gtc = GroundTruthCollector(config)
    gt_path = os.path.join(config["paths"]["ground_truth"], "ground_truth_data.json")
    if os.path.exists(gt_path):
        gt_data = gtc.load_ground_truth(gt_path)
    else:
        gt_data = gtc.extract_from_queries(queries)

    # Run NLP evaluation
    evaluator = NLPEvaluator(config)
    evaluations = evaluator.batch_evaluate(responses, gt_data, queries)

    # Save evaluations
    eval_output = os.path.join(config["paths"]["results"], "nlp_evaluations.json")
    os.makedirs(os.path.dirname(eval_output), exist_ok=True)
    with open(eval_output, "w") as f:
        json.dump(evaluations, f, indent=2, default=str)
    print(f"\nNLP evaluations saved to {eval_output}")

    # Score evaluations
    scorer = Scorer(config)
    scores_df = scorer.score_all(evaluations, queries, responses)
    summary = scorer.compute_summary_statistics(scores_df)
    domain_breakdown = scorer.compute_domain_breakdown(scores_df)

    print("\n--- Summary Statistics ---")
    print(summary.to_string())

    return scores_df


def run_analyze(config: dict):
    """Stage 3: Statistical analysis and visualization."""
    print("\n" + "=" * 60)
    print("STAGE 3: STATISTICAL ANALYSIS & VISUALIZATION")
    print("=" * 60)

    # Load scores
    scores_path = os.path.join(config["paths"]["results"], "evaluation_scores.csv")
    if not os.path.exists(scores_path):
        print(f"ERROR: Scores file not found: {scores_path}")
        print("Run evaluation first: python main.py --mode evaluate")
        sys.exit(1)

    import pandas as pd
    scores_df = pd.read_csv(scores_path)
    print(f"Loaded {len(scores_df)} evaluation scores")

    # Statistical analysis
    analyzer = StatisticalAnalyzer(config)
    stat_results = analyzer.run_full_analysis(scores_df)
    analyzer.print_summary(stat_results)

    # Visualization
    visualizer = Visualizer(config)
    visualizer.generate_all(scores_df, stat_results)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults directory: {config['paths']['results']}")
    print(f"Figures directory: {config['paths']['figures']}")


def run_demo(config: dict):
    """Run a demo with simulated data to showcase the pipeline."""
    print("\n" + "=" * 60)
    print("DEMO MODE: Running with Simulated Data")
    print("=" * 60)

    qm = QueryManager(config["paths"]["queries"])
    queries = qm.get_all_queries()

    # Generate simulated responses
    print("\nGenerating simulated chatbot responses...")
    import random
    random.seed(42)

    responses = {}
    chatbot_ids = ["chatgpt", "gemini", "claude", "grok", "deepseek"]

    for query in queries:
        qid = query["id"]
        gt = query.get("ground_truth", "")
        responses[qid] = {}

        for cb_id in chatbot_ids:
            # Simulate responses with varying quality
            # Each chatbot has different "personality" in errors
            noise_level = {
                "chatgpt": 0.15,
                "gemini": 0.18,
                "claude": 0.12,
                "grok": 0.20,
                "deepseek": 0.22,
            }[cb_id]

            # Create a response based on ground truth with some variation
            words = gt.split()
            if random.random() < noise_level:
                # Introduce some errors
                if len(words) > 5:
                    idx = random.randint(0, len(words) - 1)
                    words[idx] = random.choice(["approximately", "roughly", "about"])

            simulated_response = " ".join(words)

            # Sometimes add extra info
            if random.random() > 0.5:
                simulated_response += " This is additional context that may or may not be accurate."

            responses[qid][cb_id] = {
                "chatbot_id": cb_id,
                "chatbot_name": cb_id.title(),
                "response": simulated_response,
                "timestamp": datetime.now().isoformat(),
                "error": None,
            }

    # Save simulated responses
    os.makedirs(config["paths"]["responses"], exist_ok=True)
    resp_path = os.path.join(config["paths"]["responses"], "chatbot_responses.json")
    with open(resp_path, "w") as f:
        json.dump(responses, f, indent=2)
    print(f"  → Simulated responses saved to {resp_path}")

    # Extract ground truth
    gtc = GroundTruthCollector(config)
    gt_data = gtc.extract_from_queries(queries)

    # Run NLP evaluation
    print("\nRunning NLP evaluation...")
    evaluator = NLPEvaluator(config)
    evaluations = evaluator.batch_evaluate(responses, gt_data, queries)

    eval_output = os.path.join(config["paths"]["results"], "nlp_evaluations.json")
    os.makedirs(os.path.dirname(eval_output), exist_ok=True)
    with open(eval_output, "w") as f:
        json.dump(evaluations, f, indent=2, default=str)

    # Score
    scorer = Scorer(config)
    scores_df = scorer.score_all(evaluations, queries, responses)
    summary = scorer.compute_summary_statistics(scores_df)
    scorer.compute_domain_breakdown(scores_df)

    print("\n--- Summary Statistics (Demo) ---")
    print(summary.to_string())

    # Statistical analysis
    analyzer = StatisticalAnalyzer(config)
    stat_results = analyzer.run_full_analysis(scores_df)

    # Visualization
    visualizer = Visualizer(config)
    visualizer.generate_all(scores_df, stat_results)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to: {config['paths']['results']}")
    print("\nNote: This used simulated data. For real results:")
    print("  1. Set up API keys in environment variables")
    print("  2. Run: python main.py --mode full")


def run_template(config: dict):
    """Generate manual collection template."""
    qm = QueryManager(config["paths"]["queries"])
    queries = qm.get_all_queries()
    ci = ChatbotInterface(config)
    ci.generate_manual_collection_template(queries)


def main():
    parser = argparse.ArgumentParser(
        description="AI Chatbot Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  full      Run complete pipeline (collect → evaluate → analyze)
  collect   Collect chatbot responses via APIs
  evaluate  Evaluate collected responses with NLP
  analyze   Run statistical analysis and generate visualizations
  demo      Run with simulated data (no API keys needed)
  template  Generate manual response collection template
        """
    )
    parser.add_argument(
        "--mode", type=str, default="demo",
        choices=["full", "collect", "evaluate", "analyze", "demo", "template"],
        help="Pipeline mode to run (default: demo)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()
    print_banner()
    config = load_config(args.config)

    if args.mode == "full":
        run_collect(config)
        run_evaluate(config)
        run_analyze(config)
    elif args.mode == "collect":
        run_collect(config)
    elif args.mode == "evaluate":
        run_evaluate(config)
    elif args.mode == "analyze":
        run_analyze(config)
    elif args.mode == "demo":
        run_demo(config)
    elif args.mode == "template":
        run_template(config)


if __name__ == "__main__":
    main()
