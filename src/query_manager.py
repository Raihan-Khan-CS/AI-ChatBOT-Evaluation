"""
Query Manager Module
====================
Handles loading, filtering, and managing the evaluation query dataset.
"""

import json
import os
from typing import List, Dict, Optional
from collections import Counter


class QueryManager:
    """Manages the evaluation query dataset."""

    def __init__(self, queries_path: str = "data/queries/evaluation_queries.json"):
        self.queries_path = queries_path
        self.queries: List[Dict] = []
        self.metadata: Dict = {}
        self._load_queries()

    def _load_queries(self):
        """Load queries from JSON file."""
        if not os.path.exists(self.queries_path):
            raise FileNotFoundError(f"Query file not found: {self.queries_path}")

        with open(self.queries_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        self.queries = data.get("queries", [])
        print(f"[QueryManager] Loaded {len(self.queries)} queries from {self.queries_path}")

    def get_all_queries(self) -> List[Dict]:
        """Return all queries."""
        return self.queries

    def get_query_by_id(self, query_id: str) -> Optional[Dict]:
        """Get a specific query by its ID."""
        for q in self.queries:
            if q["id"] == query_id:
                return q
        return None

    def filter_by_domain(self, domain: str) -> List[Dict]:
        """Filter queries by domain (e.g., 'Science', 'History')."""
        return [q for q in self.queries if q["domain"].lower() == domain.lower()]

    def filter_by_type(self, query_type: str) -> List[Dict]:
        """Filter queries by type (e.g., 'Factual', 'Comparative')."""
        return [q for q in self.queries if q["type"].lower() == query_type.lower()]

    def filter_by_difficulty(self, difficulty: str) -> List[Dict]:
        """Filter queries by difficulty level."""
        return [q for q in self.queries if q.get("difficulty", "").lower() == difficulty.lower()]

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        domains = Counter(q["domain"] for q in self.queries)
        types = Counter(q["type"] for q in self.queries)
        difficulties = Counter(q.get("difficulty", "Unknown") for q in self.queries)

        return {
            "total_queries": len(self.queries),
            "domains": dict(domains),
            "types": dict(types),
            "difficulties": dict(difficulties),
        }

    def get_questions_only(self) -> List[str]:
        """Return just the question strings."""
        return [q["question"] for q in self.queries]

    def export_for_chatbot(self, output_path: str = None) -> List[Dict]:
        """Export queries in a format ready for chatbot submission."""
        exported = []
        for q in self.queries:
            exported.append({
                "id": q["id"],
                "question": q["question"],
                "domain": q["domain"],
                "type": q["type"],
            })

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(exported, f, indent=2)
            print(f"[QueryManager] Exported {len(exported)} queries to {output_path}")

        return exported

    def __len__(self):
        return len(self.queries)

    def __repr__(self):
        return f"QueryManager(queries={len(self.queries)}, path='{self.queries_path}')"
