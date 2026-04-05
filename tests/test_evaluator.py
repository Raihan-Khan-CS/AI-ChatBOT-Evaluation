"""
Tests for the AI Chatbot Evaluation Framework
"""

import json
import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.query_manager import QueryManager
from src.nlp_evaluator import NLPEvaluator
from src.scorer import Scorer


class TestQueryManager:
    """Tests for QueryManager."""

    def test_load_queries(self):
        qm = QueryManager("data/queries/evaluation_queries.json")
        assert len(qm) > 0
        assert len(qm.get_all_queries()) > 0

    def test_filter_by_domain(self):
        qm = QueryManager("data/queries/evaluation_queries.json")
        science = qm.filter_by_domain("Science")
        assert all(q["domain"] == "Science" for q in science)

    def test_filter_by_type(self):
        qm = QueryManager("data/queries/evaluation_queries.json")
        factual = qm.filter_by_type("Factual")
        assert all(q["type"] == "Factual" for q in factual)

    def test_get_query_by_id(self):
        qm = QueryManager("data/queries/evaluation_queries.json")
        query = qm.get_query_by_id("SCI_F_001")
        assert query is not None
        assert query["domain"] == "Science"

    def test_statistics(self):
        qm = QueryManager("data/queries/evaluation_queries.json")
        stats = qm.get_statistics()
        assert "total_queries" in stats
        assert stats["total_queries"] > 0


class TestNLPEvaluator:
    """Tests for NLPEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return NLPEvaluator()

    def test_keyword_similarity(self, evaluator):
        score = evaluator.compute_keyword_similarity(
            "The speed of light is 299792458 meters per second.",
            "Light travels at approximately 299,792,458 m/s in vacuum."
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Should have decent similarity

    def test_keyword_similarity_unrelated(self, evaluator):
        score = evaluator.compute_keyword_similarity(
            "The cat sat on the mat.",
            "Quantum mechanics describes subatomic particles."
        )
        assert score < 0.3  # Should be low

    def test_extract_keywords(self, evaluator):
        keywords = evaluator.extract_keywords("Machine learning is a subset of artificial intelligence.")
        assert len(keywords) > 0
        assert all(isinstance(kw, tuple) and len(kw) == 2 for kw in keywords)

    def test_entity_extraction(self, evaluator):
        entities = evaluator.extract_entities(
            "Neil Armstrong walked on the Moon on July 20, 1969, during Apollo 11."
        )
        assert len(entities) > 0
        entity_texts = [e["text"] for e in entities]
        assert any("Armstrong" in t or "Neil" in t for t in entity_texts)

    def test_semantic_similarity(self, evaluator):
        score = evaluator.compute_sentence_similarity(
            "The capital of France is Paris.",
            "Paris is the capital city of France."
        )
        assert score > 0.7  # Should be very similar

    def test_evaluate_response(self, evaluator):
        result = evaluator.evaluate_response(
            "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
            "The boiling point of water is 100°C (212°F) at 1 atmosphere."
        )
        assert "keyword_similarity" in result
        assert "entity_comparison" in result
        assert "semantic_similarity" in result
        assert "overall_nlp_score" in result
        assert result["overall_nlp_score"] > 0

    def test_empty_response(self, evaluator):
        result = evaluator.evaluate_response("", "Some ground truth.")
        assert result["overall_nlp_score"] == 0.0


class TestScorer:
    """Tests for Scorer."""

    def test_nlp_to_likert(self):
        scorer = Scorer()
        assert scorer._nlp_to_likert(0.0) == 1
        assert scorer._nlp_to_likert(0.1) == 1
        assert scorer._nlp_to_likert(0.3) == 2
        assert scorer._nlp_to_likert(0.5) == 3
        assert scorer._nlp_to_likert(0.7) == 4
        assert scorer._nlp_to_likert(0.9) == 5
        assert scorer._nlp_to_likert(1.0) == 5

    def test_accuracy_score(self):
        scorer = Scorer()
        evaluation = {
            "keyword_similarity": 0.8,
            "keyword_overlap": {"overlap": 0.7},
            "entity_comparison": {"entity_f1": 0.6},
            "semantic_similarity": 0.85,
        }
        result = scorer.compute_accuracy_score(evaluation)
        assert "raw_score" in result
        assert "likert_score" in result
        assert 1 <= result["likert_score"] <= 5
        assert 0.0 <= result["raw_score"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
