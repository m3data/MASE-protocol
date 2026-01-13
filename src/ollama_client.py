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
import threading
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set


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
        seed: Optional[int] = None,
        extra_options: Optional[Dict[str, float]] = None
    ) -> Tuple[str, ResponseMetadata]:
        """
        Generate a response from the specified model.

        Args:
            model: Ollama model name (e.g., "llama3:latest", "phi3:latest")
            messages: Chat messages in OpenAI format [{"role": "...", "content": "..."}]
            temperature: Sampling temperature (0.0-1.0)
            seed: Random seed for reproducibility (if supported by model)
            extra_options: Additional Ollama options (top_p, repeat_penalty, etc.)

        Returns:
            Tuple of (response_text, metadata)
        """
        url = f"{self.base_url}/api/chat"

        options = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed
        if extra_options:
            options.update(extra_options)

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

    def warm_model(self, model: str) -> bool:
        """
        Send a minimal request to keep a model loaded in memory.

        Args:
            model: Model name to warm

        Returns:
            True if successful, False otherwise
        """
        try:
            # Minimal request to touch the model without expensive inference
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": "hi",
                "options": {"num_predict": 1}  # Generate just 1 token
            }
            response = requests.post(url, json=payload, timeout=30)
            return response.status_code == 200
        except:
            return False


class ModelWarmthManager:
    """
    Keeps models loaded in Ollama memory during long-running experiments.

    Ollama unloads models after ~5 minutes of inactivity. For experiments
    spanning hours with multiple models, this causes expensive reloads
    mid-run. This manager periodically pings models to keep them hot.
    """

    def __init__(
        self,
        client: OllamaClient,
        models: List[str],
        interval_seconds: int = 180  # 3 minutes (well under 5-min unload)
    ):
        """
        Initialize warmth manager.

        Args:
            client: OllamaClient instance
            models: List of models to keep warm
            interval_seconds: How often to ping (default 3 minutes)
        """
        self.client = client
        self.models: Set[str] = set(models)
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_warm: Dict[str, float] = {}

    def start(self):
        """Start the background warmth maintenance thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # Already running

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._warmth_loop, daemon=True)
        self._thread.start()
        print(f"  [ModelWarmth] Started for {len(self.models)} models (interval: {self.interval}s)")

    def stop(self):
        """Stop the background warmth thread."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None
        print("  [ModelWarmth] Stopped")

    def touch(self, model: str):
        """Record that a model was just used (resets its warmth timer)."""
        self._last_warm[model] = time.time()

    def _warmth_loop(self):
        """Background loop that periodically warms idle models."""
        while not self._stop_event.wait(timeout=self.interval):
            now = time.time()
            for model in self.models:
                # Skip if recently used or warmed
                last = self._last_warm.get(model, 0)
                if now - last < self.interval:
                    continue

                # Warm the model
                success = self.client.warm_model(model)
                if success:
                    self._last_warm[model] = now
                else:
                    print(f"  [ModelWarmth] Warning: failed to warm {model}")


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
