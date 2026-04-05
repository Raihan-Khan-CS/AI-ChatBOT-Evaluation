"""
Ground Truth Collector Module
=============================
Collects and manages ground truth data from authoritative sources
for validating chatbot responses.
"""

import json
import os
import requests
from datetime import datetime
from typing import List, Dict, Optional
from bs4 import BeautifulSoup


class GroundTruthCollector:
    """Collects ground truth data from authoritative sources."""

    def __init__(self, config: Dict):
        self.config = config
        self.gt_dir = config.get("paths", {}).get("ground_truth", "data/ground_truth/")
        os.makedirs(self.gt_dir, exist_ok=True)
        self.headers = {"User-Agent": "Mozilla/5.0 (Research Bot - Academic Project)"}

    def extract_from_queries(self, queries: List[Dict]) -> Dict:
        """
        Extract ground truth data already embedded in the query dataset.

        Each query in our dataset already contains ground_truth,
        ground_truth_source, and ground_truth_date fields.
        """
        ground_truth_data = {}

        for query in queries:
            ground_truth_data[query["id"]] = {
                "query_id": query["id"],
                "question": query["question"],
                "domain": query["domain"],
                "type": query["type"],
                "ground_truth": query.get("ground_truth", ""),
                "source": query.get("ground_truth_source", "Unknown"),
                "date": query.get("ground_truth_date", ""),
                "verified": True,
                "collection_method": "pre-curated",
            }

        # Save to file
        output_path = os.path.join(self.gt_dir, "ground_truth_data.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth_data, f, indent=2, ensure_ascii=False)

        print(f"[GroundTruthCollector] Extracted {len(ground_truth_data)} ground truth entries")
        print(f"  → Saved to {output_path}")

        return ground_truth_data

    def fetch_wikipedia_summary(self, topic: str) -> Optional[str]:
        """Fetch a summary from Wikipedia API."""
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_")
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                return data.get("extract", "")
        except requests.RequestException as e:
            print(f"  [Wikipedia] Error fetching '{topic}': {e}")
        return None

    def scrape_webpage(self, url: str) -> Optional[str]:
        """Scrape text content from a webpage."""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            text = soup.get_text(separator=" ", strip=True)
            # Clean up whitespace
            text = " ".join(text.split())
            return text[:5000]  # Limit length
        except requests.RequestException as e:
            print(f"  [Scraper] Error fetching '{url}': {e}")
        return None

    def enrich_ground_truth(self, ground_truth_data: Dict, queries: List[Dict]) -> Dict:
        """
        Optionally enrich ground truth data with additional web sources.

        This adds supplementary information from Wikipedia to existing
        ground truth entries.
        """
        print("\n[GroundTruthCollector] Enriching ground truth with Wikipedia data...")

        for query in queries:
            qid = query["id"]
            if qid not in ground_truth_data:
                continue

            # Try to get Wikipedia context for non-current-events queries
            if query["domain"] != "Current Events":
                # Extract key terms from the question for Wikipedia lookup
                key_terms = self._extract_key_terms(query["question"])
                for term in key_terms[:2]:  # Try first 2 terms
                    wiki_text = self.fetch_wikipedia_summary(term)
                    if wiki_text:
                        ground_truth_data[qid]["wikipedia_context"] = wiki_text
                        ground_truth_data[qid]["wikipedia_term"] = term
                        break

        # Save enriched data
        output_path = os.path.join(self.gt_dir, "ground_truth_enriched.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth_data, f, indent=2, ensure_ascii=False)

        enriched_count = sum(
            1 for v in ground_truth_data.values()
            if "wikipedia_context" in v
        )
        print(f"  → Enriched {enriched_count}/{len(ground_truth_data)} entries with Wikipedia context")

        return ground_truth_data

    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract potential Wikipedia search terms from a question."""
        # Remove common question words
        stop_words = {
            "what", "is", "are", "was", "were", "who", "when", "where",
            "how", "why", "which", "the", "a", "an", "in", "of", "and",
            "to", "for", "does", "do", "did", "has", "have", "had",
            "can", "could", "would", "should", "its", "it", "that",
            "this", "as", "at", "by", "from", "on", "with", "between",
            "many", "much", "most", "current", "latest", "recent",
        }

        words = question.lower().rstrip("?").split()
        # Look for capitalized multi-word terms in original
        original_words = question.rstrip("?").split()

        # Find proper nouns (capitalized words)
        terms = []
        i = 0
        while i < len(original_words):
            if original_words[i][0].isupper() and original_words[i].lower() not in stop_words:
                # Try to capture multi-word proper nouns
                phrase = [original_words[i]]
                j = i + 1
                while j < len(original_words) and original_words[j][0].isupper():
                    phrase.append(original_words[j])
                    j += 1
                terms.append(" ".join(phrase))
                i = j
            else:
                i += 1

        # Fallback: use longest non-stop words
        if not terms:
            filtered = [w for w in words if w not in stop_words and len(w) > 3]
            terms = sorted(filtered, key=len, reverse=True)[:3]

        return terms

    def load_ground_truth(self, filepath: str = None) -> Dict:
        """Load ground truth data from file."""
        if filepath is None:
            filepath = os.path.join(self.gt_dir, "ground_truth_data.json")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Ground truth file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_statistics(self, ground_truth_data: Dict) -> Dict:
        """Get statistics about the ground truth dataset."""
        sources = {}
        domains = {}

        for entry in ground_truth_data.values():
            src = entry.get("source", "Unknown")
            sources[src] = sources.get(src, 0) + 1
            dom = entry.get("domain", "Unknown")
            domains[dom] = domains.get(dom, 0) + 1

        return {
            "total_entries": len(ground_truth_data),
            "sources": sources,
            "domains": domains,
            "enriched_count": sum(
                1 for v in ground_truth_data.values()
                if "wikipedia_context" in v
            ),
        }
