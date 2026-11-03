"""Prime inference transport; credentials never enter experiment artifacts."""

import json
import math
import os
import re
import urllib.error
import urllib.request


class ChatEditor:
    """One JSON tool action per model turn, also used by the Verifiers environment.

    Credentials come from environment variables only. A custom URL is useful for
    evaluating exported trained weights served by a separate vLLM instance.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.pinference.ai/api/v1",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        input_price: float | None = None,
        output_price: float | None = None,
        max_cost_usd: float = 1.0,
        request=None,
        json_mode: bool = False,
        base_model: str | None = None,
        request_timeout: float = 90,
    ):
        if (
            input_price is None
            or output_price is None
            or not all(math.isfinite(p) and p >= 0 for p in (input_price, output_price))
        ):
            raise ValueError("Explicit current input/output prices per million tokens are required")
        if not math.isfinite(max_cost_usd) or max_cost_usd <= 0 or max_tokens <= 0:
            raise ValueError("A positive inference spending limit is required")
        self.model, self.base_url = model, base_url.rstrip("/")
        self.max_tokens, self.temperature = max_tokens, temperature
        self.input_price, self.output_price = input_price, output_price
        self.max_cost_usd, self.spent = max_cost_usd, 0.0
        self.request = request or self._request
        self.json_mode = json_mode
        self.base_model = base_model or model
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("request_timeout must be positive and finite")
        self.request_timeout = request_timeout

    def _request(self, payload: dict) -> dict:
        key = os.environ.get("PRIME_API_KEY")
        if not key:
            try:
                import httpx
                from prime_cli.api.inference import InferenceAPIError, InferenceClient
            except ImportError:
                raise RuntimeError(
                    "Set PRIME_API_KEY or install the prime extra and log in"
                ) from None

            try:
                # Use the official CLI's credential loader without copying or logging secrets.
                client = InferenceClient(timeout=self.request_timeout, inference_url=self.base_url)
                try:
                    return client.chat_completion(payload)
                finally:
                    client._client.close()
            except httpx.TimeoutException:
                raise RuntimeError("Prime inference request timed out") from None
            except (InferenceAPIError, httpx.HTTPError, OSError, ValueError) as exc:
                match = re.search(r"failed:\s*(\d{3})", str(exc))
                if match:
                    raise RuntimeError(f"Prime inference HTTP {match.group(1)}") from None
                raise RuntimeError(
                    "Prime inference unavailable; check CLI login and permissions"
                ) from None
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if os.environ.get("PRIME_TEAM_ID"):
            headers["X-Prime-Team-ID"] = os.environ["PRIME_TEAM_ID"]
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(), headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # Provider response bodies can contain private request details.
            raise RuntimeError(f"Inference HTTP {exc.code}; response body omitted") from None
