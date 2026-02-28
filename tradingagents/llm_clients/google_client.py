import logging
import os
import random
import time
from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI

from .base_client import BaseLLMClient
from .validators import validate_model

logger = logging.getLogger(__name__)

TRANSIENT_NETWORK_ERROR_TYPE_NAMES = {
    "ConnectError",
    "ConnectTimeout",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "WriteError",
    "WriteTimeout",
    "PoolTimeout",
    "ProtocolError",
    "ConnectionResetError",
    "ConnectionError",
    "TimeoutError",
    "ServiceUnavailable",
    "TooManyRequests",
    "InternalServerError",
    "DeadlineExceeded",
}
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _parse_bool(value: Any) -> Optional[bool]:
    """Convert common truthy/falsy env/config values to bool."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _parse_int(value: Any, default: int, minimum: int = 0) -> int:
    """Parse integer from env/config value with safe fallback."""
    if value is None:
        return default
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def _parse_float(value: Any, default: float, minimum: float = 0.0) -> float:
    """Parse float from env/config value with safe fallback."""
    if value is None:
        return default
    try:
        return max(minimum, float(value))
    except (TypeError, ValueError):
        return default


def _iter_exception_chain(error: BaseException):
    """Yield an exception and chained causes/contexts."""
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = current.__cause__ or current.__context__


def _is_transient_network_error(error: BaseException) -> bool:
    """Check whether an exception is a retryable transient network/server error."""
    for exc in _iter_exception_chain(error):
        if exc.__class__.__name__ in TRANSIENT_NETWORK_ERROR_TYPE_NAMES:
            return True

        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and status_code in TRANSIENT_STATUS_CODES:
            return True

        message = str(exc).lower()
        if any(
            text in message
            for text in (
                "server disconnected without sending a response",
                "connection reset",
                "temporarily unavailable",
                "timed out",
                "timeout",
                "broken pipe",
            )
        ):
            return True

    return False


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with normalized content output.

    Gemini 3 models return content as list: [{'type': 'text', 'text': '...'}]
    This normalizes to string for consistent downstream handling.
    """

    def __init__(self, **kwargs):
        # Extra retry config for transient network errors on top of SDK retries.
        network_max_retries = _parse_int(
            kwargs.pop(
                "network_max_retries",
                os.getenv("TRADINGAGENTS_GOOGLE_NETWORK_MAX_RETRIES", 4),
            ),
            default=4,
            minimum=0,
        )
        network_retry_base_delay = _parse_float(
            kwargs.pop(
                "network_retry_base_delay",
                os.getenv("TRADINGAGENTS_GOOGLE_NETWORK_RETRY_BASE_DELAY", 1.0),
            ),
            default=1.0,
            minimum=0.0,
        )
        network_retry_max_delay = _parse_float(
            kwargs.pop(
                "network_retry_max_delay",
                os.getenv("TRADINGAGENTS_GOOGLE_NETWORK_RETRY_MAX_DELAY", 12.0),
            ),
            default=12.0,
            minimum=0.0,
        )
        network_retry_jitter = _parse_float(
            kwargs.pop(
                "network_retry_jitter",
                os.getenv("TRADINGAGENTS_GOOGLE_NETWORK_RETRY_JITTER", 0.5),
            ),
            default=0.5,
            minimum=0.0,
        )

        super().__init__(**kwargs)
        object.__setattr__(self, "_network_max_retries", network_max_retries)
        object.__setattr__(self, "_network_retry_base_delay", network_retry_base_delay)
        object.__setattr__(self, "_network_retry_max_delay", network_retry_max_delay)
        object.__setattr__(self, "_network_retry_jitter", network_retry_jitter)

    def _normalize_content(self, response):
        content = response.content
        if isinstance(content, list):
            texts = [
                item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
                else item if isinstance(item, str) else ""
                for item in content
            ]
            response.content = "\n".join(t for t in texts if t)
        return response

    def invoke(self, input, config=None, **kwargs):
        total_attempts = self._network_max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                return self._normalize_content(super().invoke(input, config, **kwargs))
            except Exception as exc:
                if attempt >= total_attempts or not _is_transient_network_error(exc):
                    raise

                backoff = min(
                    self._network_retry_max_delay,
                    self._network_retry_base_delay * (2 ** (attempt - 1)),
                )
                if self._network_retry_jitter > 0:
                    backoff += random.uniform(0, self._network_retry_jitter)

                logger.warning(
                    "Transient Gemini network error (%s). Retry %d/%d in %.2fs.",
                    exc.__class__.__name__,
                    attempt,
                    self._network_max_retries,
                    backoff,
                )
                time.sleep(backoff)


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatGoogleGenerativeAI instance."""
        llm_kwargs = {"model": self.model}

        # Vertex AI mode is controlled via env vars in google-genai SDK.
        # Keep compatibility with both direct Gemini API and Vertex AI.
        use_vertexai = _parse_bool(
            self.kwargs.get("use_vertexai", os.getenv("GOOGLE_GENAI_USE_VERTEXAI"))
        )
        google_cloud_project = self.kwargs.get("google_cloud_project") or os.getenv(
            "GOOGLE_CLOUD_PROJECT"
        )
        google_cloud_location = self.kwargs.get("google_cloud_location") or os.getenv(
            "GOOGLE_CLOUD_LOCATION"
        )

        if use_vertexai is not None:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = (
                "true" if use_vertexai else "false"
            )
        if google_cloud_project:
            os.environ["GOOGLE_CLOUD_PROJECT"] = google_cloud_project
        if google_cloud_location:
            os.environ["GOOGLE_CLOUD_LOCATION"] = google_cloud_location

        for key in (
            "timeout",
            "max_retries",
            "google_api_key",
            "callbacks",
            "network_max_retries",
            "network_retry_base_delay",
            "network_retry_max_delay",
            "network_retry_jitter",
        ):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Map thinking_level to appropriate API param based on model
        # Gemini 3 Pro: low, high
        # Gemini 3 Flash: minimal, low, medium, high
        # Gemini 2.5: thinking_budget (0=disable, -1=dynamic)
        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            model_lower = self.model.lower()
            if "gemini-3" in model_lower:
                # Gemini 3 Pro doesn't support "minimal", use "low" instead
                if "pro" in model_lower and thinking_level == "minimal":
                    thinking_level = "low"
                llm_kwargs["thinking_level"] = thinking_level
            else:
                # Gemini 2.5: map to thinking_budget
                llm_kwargs["thinking_budget"] = -1 if thinking_level == "high" else 0

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Google."""
        return validate_model("google", self.model)
