"""
Chatbot Interface Module
========================
Provides a unified interface for querying multiple AI chatbots
and collecting their responses.
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm


class ChatbotInterface:
    """Unified interface for querying AI chatbots."""

    SUPPORTED_CHATBOTS = ["chatgpt", "gemini", "claude", "grok", "deepseek"]

    def __init__(self, config: Dict):
        self.config = config
        self.chatbots = {cb["id"]: cb for cb in config.get("chatbots", [])}
        self.responses_dir = config.get("paths", {}).get("responses", "data/responses/")
        os.makedirs(self.responses_dir, exist_ok=True)

    def _get_api_key(self, chatbot_id: str) -> Optional[str]:
        """Retrieve API key from environment variables."""
        cb = self.chatbots.get(chatbot_id, {})
        env_var = cb.get("api_key_env", "")
        return os.environ.get(env_var)

    def query_chatgpt(self, question: str, api_key: str) -> str:
        """Query OpenAI ChatGPT API."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.chatbots["chatgpt"].get("model", "gpt-4o"),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Provide accurate, factual answers."},
                {"role": "user", "content": question},
            ],
            "max_tokens": 1000,
            "temperature": 0.1,
        }
        response = requests.post(
            self.chatbots["chatgpt"]["endpoint"],
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def query_gemini(self, question: str, api_key: str) -> str:
        """Query Google Gemini API."""
        model = self.chatbots["gemini"].get("model", "gemini-pro")
        url = f"{self.chatbots['gemini']['endpoint']}/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": question}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1000},
        }
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]

    def query_claude(self, question: str, api_key: str) -> str:
        """Query Anthropic Claude API."""
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.chatbots["claude"].get("model", "claude-sonnet-4-20250514"),
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": question}],
        }
        response = requests.post(
            self.chatbots["claude"]["endpoint"],
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]

    def query_grok(self, question: str, api_key: str) -> str:
        """Query xAI Grok API."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.chatbots["grok"].get("model", "grok-2"),
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 1000,
            "temperature": 0.1,
        }
        response = requests.post(
            self.chatbots["grok"]["endpoint"],
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def query_deepseek(self, question: str, api_key: str) -> str:
        """Query DeepSeek API."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.chatbots["deepseek"].get("model", "deepseek-chat"),
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            "max_tokens": 1000,
            "temperature": 0.1,
        }
        response = requests.post(
            self.chatbots["deepseek"]["endpoint"],
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def query_single(self, chatbot_id: str, question: str) -> Dict:
        """Query a single chatbot and return structured response."""
        api_key = self._get_api_key(chatbot_id)
        timestamp = datetime.now().isoformat()

        result = {
            "chatbot_id": chatbot_id,
            "chatbot_name": self.chatbots[chatbot_id]["name"],
            "question": question,
            "timestamp": timestamp,
            "response": None,
            "error": None,
            "response_time_ms": None,
        }

        if not api_key:
            result["error"] = f"API key not found for {chatbot_id} (env: {self.chatbots[chatbot_id].get('api_key_env')})"
            return result

        query_methods = {
            "chatgpt": self.query_chatgpt,
            "gemini": self.query_gemini,
            "claude": self.query_claude,
            "grok": self.query_grok,
            "deepseek": self.query_deepseek,
        }

        try:
            start = time.time()
            response = query_methods[chatbot_id](question, api_key)
            elapsed = (time.time() - start) * 1000
            result["response"] = response
            result["response_time_ms"] = round(elapsed, 2)
        except Exception as e:
            result["error"] = str(e)

        return result

    def query_all_chatbots(self, question: str, delay: float = 1.0) -> List[Dict]:
        """Query all enabled chatbots with the same question."""
        results = []
        for cb_id, cb_config in self.chatbots.items():
            if not cb_config.get("enabled", True):
                continue
            result = self.query_single(cb_id, question)
            results.append(result)
            time.sleep(delay)  # Rate limiting
        return results

    def collect_all_responses(self, queries: List[Dict], delay: float = 1.5) -> Dict:
        """
        Collect responses from all chatbots for all queries.

        Returns a dictionary: {query_id: {chatbot_id: response_dict}}
        """
        all_responses = {}

        print(f"\n{'='*60}")
        print("Collecting Chatbot Responses")
        print(f"{'='*60}")
        print(f"Queries: {len(queries)} | Chatbots: {len(self.chatbots)}")
        print(f"{'='*60}\n")

        for query in tqdm(queries, desc="Processing queries"):
            query_id = query["id"]
            question = query["question"]
            all_responses[query_id] = {}

            for cb_id, cb_config in self.chatbots.items():
                if not cb_config.get("enabled", True):
                    continue
                result = self.query_single(cb_id, question)
                all_responses[query_id][cb_id] = result
                time.sleep(delay)

        # Save responses
        self._save_responses(all_responses)
        return all_responses

    def _save_responses(self, responses: Dict):
        """Save collected responses to JSON."""
        output_path = os.path.join(self.responses_dir, "chatbot_responses.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)
        print(f"\n[ChatbotInterface] Responses saved to {output_path}")

    def load_responses(self, filepath: str = None) -> Dict:
        """Load previously collected responses."""
        if filepath is None:
            filepath = os.path.join(self.responses_dir, "chatbot_responses.json")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Responses file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_manual_responses(self, filepath: str) -> Dict:
        """
        Load manually collected chatbot responses from a JSON file.

        This is used when API access is unavailable and responses were
        collected by manually querying each chatbot's web interface.

        Expected format:
        {
            "query_id": {
                "chatbot_id": {
                    "response": "...",
                    "timestamp": "..."
                }
            }
        }
        """
        with open(filepath, "r", encoding="utf-8") as f:
            manual_data = json.load(f)

        # Normalize to standard format
        responses = {}
        for query_id, chatbot_responses in manual_data.items():
            responses[query_id] = {}
            for cb_id, data in chatbot_responses.items():
                responses[query_id][cb_id] = {
                    "chatbot_id": cb_id,
                    "chatbot_name": self.chatbots.get(cb_id, {}).get("name", cb_id),
                    "response": data.get("response", ""),
                    "timestamp": data.get("timestamp", datetime.now().isoformat()),
                    "error": None,
                    "response_time_ms": None,
                }

        return responses

    def generate_manual_collection_template(self, queries: List[Dict], output_path: str = None):
        """
        Generate a template JSON file for manual response collection.

        Use this when API access is unavailable - manually fill in responses
        by querying each chatbot's web interface.
        """
        if output_path is None:
            output_path = os.path.join(self.responses_dir, "manual_collection_template.json")

        template = {}
        for query in queries:
            template[query["id"]] = {
                "_question": query["question"],
                "_domain": query["domain"],
                "_type": query["type"],
            }
            for cb_id in self.chatbots:
                template[query["id"]][cb_id] = {
                    "response": "<PASTE RESPONSE HERE>",
                    "timestamp": "<YYYY-MM-DDTHH:MM:SS>",
                }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(template, f, indent=2, ensure_ascii=False)

        print(f"[ChatbotInterface] Manual collection template saved to {output_path}")
        print(f"  → Fill in responses by querying each chatbot manually")
        print(f"  → Then load with: interface.load_manual_responses('{output_path}')")

        return output_path
