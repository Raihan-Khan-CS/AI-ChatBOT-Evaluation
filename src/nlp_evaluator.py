"""
NLP Evaluator Module
====================
Core evaluation engine using NLP techniques to compare chatbot
responses against ground truth data.

Techniques:
- TF-IDF Keyword Matching
- Named Entity Recognition (NER) with SpaCy
- Semantic Sentence Similarity with Sentence-BERT
- Fact extraction and comparison
"""

import re
import warnings
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)


class NLPEvaluator:
    """NLP-based evaluation engine for chatbot response quality assessment."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        eval_config = self.config.get("evaluation", {})
        self.similarity_model_name = eval_config.get("similarity_model", "all-MiniLM-L6-v2")
        self.spacy_model_name = eval_config.get("spacy_model", "en_core_web_sm")

        # Lazy-loaded models
        self._nlp = None
        self._sentence_model = None
        self._tfidf_vectorizer = None

        print("[NLPEvaluator] Initialized (models loaded on first use)")

    @property
    def nlp(self):
        """Lazy-load SpaCy model."""
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load(self.spacy_model_name)
                print(f"  → SpaCy model '{self.spacy_model_name}' loaded")
            except OSError:
                print(f"  → SpaCy model '{self.spacy_model_name}' not found. Downloading...")
                from spacy.cli import download
                download(self.spacy_model_name)
                self._nlp = spacy.load(self.spacy_model_name)
        return self._nlp

    @property
    def sentence_model(self):
        """Lazy-load Sentence-BERT model."""
        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer
            self._sentence_model = SentenceTransformer(self.similarity_model_name)
            print(f"  → Sentence-BERT model '{self.similarity_model_name}' loaded")
        return self._sentence_model

    # ----------------------------------------------------------------
    # 1. TF-IDF Keyword Matching
    # ----------------------------------------------------------------

    def compute_keyword_similarity(self, text1: str, text2: str) -> float:
        """
        Compute TF-IDF based cosine similarity between two texts.

        Returns a score between 0.0 and 1.0.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        if not text1 or not text2:
            return 0.0

        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except ValueError:
            return 0.0

    def extract_keywords(self, text: str, top_n: int = 15) -> List[Tuple[str, float]]:
        """Extract top keywords using TF-IDF scores."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not text:
            return []

        vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            return keyword_scores[:top_n]
        except ValueError:
            return []

    def compute_keyword_overlap(self, response: str, ground_truth: str) -> Dict:
        """Compute keyword overlap between response and ground truth."""
        resp_keywords = set(kw for kw, _ in self.extract_keywords(response, top_n=20))
        gt_keywords = set(kw for kw, _ in self.extract_keywords(ground_truth, top_n=20))

        if not gt_keywords:
            return {"overlap": 0.0, "precision": 0.0, "recall": 0.0, "common": [], "missing": []}

        common = resp_keywords & gt_keywords
        precision = len(common) / len(resp_keywords) if resp_keywords else 0.0
        recall = len(common) / len(gt_keywords) if gt_keywords else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "overlap": f1,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "common_keywords": sorted(common),
            "missing_keywords": sorted(gt_keywords - resp_keywords),
        }

    # ----------------------------------------------------------------
    # 2. Named Entity Recognition (NER)
    # ----------------------------------------------------------------

    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities using SpaCy."""
        if not text:
            return []

        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
        return entities

    def compare_entities(self, response: str, ground_truth: str) -> Dict:
        """
        Compare named entities between response and ground truth.

        Returns entity overlap metrics and details.
        """
        resp_entities = self.extract_entities(response)
        gt_entities = self.extract_entities(ground_truth)

        # Normalize entity texts for comparison
        resp_entity_texts = set(e["text"].lower().strip() for e in resp_entities)
        gt_entity_texts = set(e["text"].lower().strip() for e in gt_entities)

        # Also compare by entity label groups
        resp_by_label = {}
        for e in resp_entities:
            resp_by_label.setdefault(e["label"], set()).add(e["text"].lower().strip())

        gt_by_label = {}
        for e in gt_entities:
            gt_by_label.setdefault(e["label"], set()).add(e["text"].lower().strip())

        # Compute overlap
        common = resp_entity_texts & gt_entity_texts
        precision = len(common) / len(resp_entity_texts) if resp_entity_texts else 0.0
        recall = len(common) / len(gt_entity_texts) if gt_entity_texts else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Label-wise comparison
        label_comparison = {}
        all_labels = set(list(resp_by_label.keys()) + list(gt_by_label.keys()))
        for label in all_labels:
            r_set = resp_by_label.get(label, set())
            g_set = gt_by_label.get(label, set())
            c_set = r_set & g_set
            label_comparison[label] = {
                "response_count": len(r_set),
                "ground_truth_count": len(g_set),
                "common_count": len(c_set),
                "common": sorted(c_set),
            }

        return {
            "entity_f1": round(f1, 4),
            "entity_precision": round(precision, 4),
            "entity_recall": round(recall, 4),
            "response_entities": len(resp_entity_texts),
            "ground_truth_entities": len(gt_entity_texts),
            "common_entities": sorted(common),
            "missing_entities": sorted(gt_entity_texts - resp_entity_texts),
            "extra_entities": sorted(resp_entity_texts - gt_entity_texts),
            "label_comparison": label_comparison,
        }

    # ----------------------------------------------------------------
    # 3. Sentence Similarity (Sentence-BERT)
    # ----------------------------------------------------------------

    def compute_sentence_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts using Sentence-BERT.

        Returns cosine similarity score between -1.0 and 1.0.
        """
        if not text1 or not text2:
            return 0.0

        embeddings = self.sentence_model.encode([text1, text2])
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def compute_sentence_level_similarity(self, response: str, ground_truth: str) -> Dict:
        """
        Compute sentence-level similarity analysis.

        Breaks texts into sentences and finds best matches.
        """
        resp_sents = self._split_sentences(response)
        gt_sents = self._split_sentences(ground_truth)

        if not resp_sents or not gt_sents:
            return {
                "average_similarity": 0.0,
                "max_similarity": 0.0,
                "min_similarity": 0.0,
                "sentence_matches": [],
            }

        # Encode all sentences
        resp_embeddings = self.sentence_model.encode(resp_sents)
        gt_embeddings = self.sentence_model.encode(gt_sents)

        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(resp_embeddings, gt_embeddings)

        # Find best match for each ground truth sentence
        matches = []
        gt_scores = []
        for j, gt_sent in enumerate(gt_sents):
            best_idx = int(np.argmax(sim_matrix[:, j]))
            best_score = float(sim_matrix[best_idx, j])
            gt_scores.append(best_score)
            matches.append({
                "ground_truth_sentence": gt_sent,
                "best_match_response": resp_sents[best_idx],
                "similarity": round(best_score, 4),
            })

        return {
            "average_similarity": round(float(np.mean(gt_scores)), 4),
            "max_similarity": round(float(np.max(gt_scores)), 4),
            "min_similarity": round(float(np.min(gt_scores)), 4),
            "coverage_above_07": round(sum(1 for s in gt_scores if s > 0.7) / len(gt_scores), 4),
            "sentence_matches": matches,
        }

    # ----------------------------------------------------------------
    # 4. Comprehensive Evaluation
    # ----------------------------------------------------------------

    def evaluate_response(self, response: str, ground_truth: str, query: Dict = None) -> Dict:
        """
        Perform comprehensive NLP evaluation of a chatbot response
        against ground truth.

        Returns detailed scores across all evaluation dimensions.
        """
        if not response or response.strip() == "":
            return {
                "keyword_similarity": 0.0,
                "keyword_overlap": {"overlap": 0.0},
                "entity_comparison": {"entity_f1": 0.0},
                "semantic_similarity": 0.0,
                "sentence_analysis": {"average_similarity": 0.0},
                "overall_nlp_score": 0.0,
                "error": "Empty response",
            }

        # 1. Keyword analysis
        keyword_sim = self.compute_keyword_similarity(response, ground_truth)
        keyword_overlap = self.compute_keyword_overlap(response, ground_truth)

        # 2. Entity comparison
        entity_comparison = self.compare_entities(response, ground_truth)

        # 3. Semantic similarity
        semantic_sim = self.compute_sentence_similarity(response, ground_truth)
        sentence_analysis = self.compute_sentence_level_similarity(response, ground_truth)

        # 4. Compute overall NLP score (weighted combination)
        overall_score = (
            0.25 * keyword_sim
            + 0.20 * keyword_overlap["overlap"]
            + 0.20 * entity_comparison["entity_f1"]
            + 0.35 * semantic_sim
        )

        return {
            "keyword_similarity": round(keyword_sim, 4),
            "keyword_overlap": keyword_overlap,
            "entity_comparison": entity_comparison,
            "semantic_similarity": round(semantic_sim, 4),
            "sentence_analysis": sentence_analysis,
            "overall_nlp_score": round(overall_score, 4),
        }

    def batch_evaluate(
        self,
        responses: Dict,
        ground_truth_data: Dict,
        queries: List[Dict],
    ) -> Dict:
        """
        Evaluate all chatbot responses against ground truth.

        Args:
            responses: {query_id: {chatbot_id: response_data}}
            ground_truth_data: {query_id: ground_truth_entry}
            queries: List of query dictionaries

        Returns:
            {query_id: {chatbot_id: evaluation_result}}
        """
        from tqdm import tqdm

        results = {}
        query_lookup = {q["id"]: q for q in queries}

        total = sum(len(cbs) for cbs in responses.values())
        pbar = tqdm(total=total, desc="Evaluating responses")

        for query_id, chatbot_responses in responses.items():
            results[query_id] = {}
            gt_entry = ground_truth_data.get(query_id, {})
            gt_text = gt_entry.get("ground_truth", "")
            query = query_lookup.get(query_id, {})

            for chatbot_id, response_data in chatbot_responses.items():
                # Skip metadata keys
                if chatbot_id.startswith("_"):
                    continue

                response_text = response_data.get("response", "")

                if response_text and response_text != "<PASTE RESPONSE HERE>":
                    evaluation = self.evaluate_response(response_text, gt_text, query)
                else:
                    evaluation = {
                        "keyword_similarity": 0.0,
                        "keyword_overlap": {"overlap": 0.0},
                        "entity_comparison": {"entity_f1": 0.0},
                        "semantic_similarity": 0.0,
                        "sentence_analysis": {"average_similarity": 0.0},
                        "overall_nlp_score": 0.0,
                        "error": "No response available",
                    }

                evaluation["chatbot_id"] = chatbot_id
                evaluation["query_id"] = query_id
                evaluation["domain"] = query.get("domain", "")
                evaluation["query_type"] = query.get("type", "")

                results[query_id][chatbot_id] = evaluation
                pbar.update(1)

        pbar.close()
        return results

    # ----------------------------------------------------------------
    # Utility Methods
    # ----------------------------------------------------------------

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using SpaCy."""
        if not text:
            return []
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences

    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        text = re.sub(r"\s+", " ", text).strip()
        text = text.lower()
        return text
