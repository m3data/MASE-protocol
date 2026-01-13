"""
Ollama API client for MASE multi-model orchestration.

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Thin wrapper around Ollama /api/chat that returns response text
with metadata (latency, tokens) for experiment tracking.
"""

import time
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class ResponseMetadata:
    """Metadata about an LLM response."""
    model: str
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class OllamaClient:
    """
    Client for Ollama local LLM server.

    Designed for MASE multi-model experiments where we need:
    - Different models for different agents
    - Response metadata for analysis
    - Deterministic seeding for reproducibility
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 600,
        max_retries: int = 3
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL
            timeout: Request timeout in seconds (default 600 for long-context inference)
            max_retries: Number of retries on timeout (default 3)
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        seed: Optional[int] = None
    ) -> Tuple[str, ResponseMetadata]:
        """
        Generate a response from the specified model.

        Args:
            model: Ollama model name (e.g., "llama3:latest", "phi3:latest")
            messages: Chat messages in OpenAI format [{"role": "...", "content": "..."}]
            temperature: Sampling temperature (0.0-1.0)
            seed: Random seed for reproducibility (if supported by model)

        Returns:
            Tuple of (response_text, metadata)
        """
        url = f"{self.base_url}/api/chat"

        options = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options
        }

        start_time = time.perf_counter()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()

                latency_ms = (time.perf_counter() - start_time) * 1000
                data = response.json()

                # Extract response text
                response_text = data.get("message", {}).get("content", "")

                # Extract token counts if available
                prompt_tokens = data.get("prompt_eval_count")
                completion_tokens = data.get("eval_count")
                total_tokens = None
                if prompt_tokens is not None and completion_tokens is not None:
                    total_tokens = prompt_tokens + completion_tokens

                metadata = ResponseMetadata(
                    model=model,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                )

                return response_text, metadata

            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"  Timeout on attempt {attempt + 1}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                continue
            except requests.exceptions.ConnectionError:
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Ensure Ollama is running: ollama serve"
                )
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Ollama API error: {e}")

        # All retries exhausted
        raise TimeoutError(f"Ollama request timed out after {self.max_retries} attempts ({self.timeout}s each)")

    @staticmethod
    def is_running(base_url: str = "http://localhost:11434") -> bool:
        """Check if Ollama server is accessible."""
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    @staticmethod
    def get_available_models(base_url: str = "http://localhost:11434") -> List[str]:
        """
        Get list of available model names.

        Returns:
            List of model names, or empty list if Ollama not running
        """
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            response.raise_for_status()

            models = response.json().get("models", [])
            return [m["name"] for m in models]
        except:
            return []

    @staticmethod
    def validate_models(
        required_models: List[str],
        base_url: str = "http://localhost:11434"
    ) -> Dict[str, bool]:
        """
        Check which required models are available.

        Args:
            required_models: List of model names to check
            base_url: Ollama server URL

        Returns:
            Dict mapping model name to availability status
        """
        available = set(OllamaClient.get_available_models(base_url))

        result = {}
        for model in required_models:
            # Check for exact match or base name match (llama3 matches llama3:latest)
            base_name = model.split(":")[0]
            result[model] = (
                model in available or
                any(m.startswith(base_name) for m in available)
            )

        return result


# Test if run directly
if __name__ == "__main__":
    print("Ollama Client Test")
    print("=" * 50)

    if OllamaClient.is_running():
        print("Ollama is running")

        models = OllamaClient.get_available_models()
        print(f"\nAvailable models: {models}")

        # Test with first available model
        if models:
            client = OllamaClient()
            test_model = models[0]
            print(f"\nTesting with {test_model}...")

            messages = [
                {"role": "user", "content": "Say hello in exactly 5 words."}
            ]

            response, meta = client.generate(test_model, messages)
            print(f"Response: {response}")
            print(f"Latency: {meta.latency_ms:.0f}ms")
            print(f"Tokens: {meta.total_tokens}")
    else:
        print("Ollama not running. Start with: ollama serve")
