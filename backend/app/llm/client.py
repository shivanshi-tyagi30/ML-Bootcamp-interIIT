"""LLM client for Ollama (native API) or vLLM (OpenAI-compatible) with schema-constrained JSON (spec Section 12).

Ollama is called through its native /api/chat endpoint rather than the OpenAI-compatible one, because only
the native API lets us set the context window (Ollama's small default silently truncates long prompts),
switch off Qwen3's "thinking" (very slow on CPU) and keep the model loaded between calls.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Protocol, TypeVar

import httpx
from pydantic import BaseModel

from app.core.text import estimate_tokens
from app.llm.json_repair import REPAIR_MESSAGE, InvalidModelOutput, parse_output

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

PROMPTS_DIR = Path(__file__).parent / "prompts"
OnRetry = Callable[[str], Awaitable[None]]
CONTEXT_BUCKETS = (8192, 16384, 32768, 65536, 131072)


def load_prompt(name: str) -> str:
    """Read a prompt file from app/llm/prompts/."""
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


class LLMUnavailable(Exception):
    """The model server could not be used (not running, model missing, timeout). Message is user-readable."""


def is_thinking_model(model: str) -> bool:
    """Models with a reasoning mode we switch off for speed."""
    return "qwen3" in model.lower()


def context_size(system: str, user: str, max_tokens: int, cap: int) -> int:
    """Context window for a call, rounded up to a few fixed sizes so Ollama rarely has to reload the model."""
    need = int(estimate_tokens(system + user) * 1.3) + max_tokens + 256
    for size in CONTEXT_BUCKETS:
        if need <= size:
            return min(size, cap)
    return cap


class JSONLLM(Protocol):
    """What the pipeline needs from an LLM; tests provide a fake."""

    async def json_call(
        self, model: str, system: str, user: str, schema: type[T], max_retries: int, max_tokens: int = 4096,
        job_id: str = "-", on_retry: OnRetry | None = None,
    ) -> T:
        """Return a validated instance of `schema`."""
        ...


class LLMClient:
    """Calls the model at temperature 0 with JSON-schema constrained output and validates it."""

    def __init__(
        self, base_url: str, backend: Literal["ollama", "vllm"], timeout: int, model_urls: dict[str, str] | None = None,
        max_context: int = 32768, keep_alive: str = "30m",
    ) -> None:
        """Create a client; `model_urls` routes specific models to another server."""
        self.backend = backend
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.routes = {m: u.rstrip("/") for m, u in (model_urls or {}).items() if u}
        self.max_context = max_context
        self.keep_alive = keep_alive

    def _url(self, model: str) -> str:
        """Server base URL for a model."""
        return self.routes.get(model, self.base_url)

    async def _ollama(self, model: str, messages: list[dict[str, str]], schema: type[BaseModel], max_tokens: int,
                      num_ctx: int) -> tuple[str, int | None]:
        """One call to Ollama's native chat API; returns (content, prompt tokens)."""
        host = self._url(model).removesuffix("/v1")
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": schema.model_json_schema(),
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "top_p": 1, "seed": 42, "num_predict": max_tokens, "num_ctx": num_ctx},
        }
        if is_thinking_model(model):
            body["think"] = False
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                r = await client.post(f"{host}/api/chat", json=body)
            except httpx.ConnectError as e:
                raise LLMUnavailable(f"Can't reach Ollama at {host}. Start the Ollama app and try again.") from e
            except httpx.TimeoutException as e:
                raise LLMUnavailable(
                    f"{model} took longer than {self.timeout} s. Use a smaller model or raise LLM_TIMEOUT_SEC.") from e
        if r.status_code == 404:
            raise LLMUnavailable(f"Model '{model}' is not downloaded in Ollama. Run: ollama pull {model}")
        if r.status_code >= 400:
            msg = r.text[:300]
            if "think" in msg.lower() and body.pop("think", None) is not None:  # older Ollama: retry without it
                return await self._ollama_plain(host, body)
            raise LLMUnavailable(f"Ollama returned HTTP {r.status_code}: {msg}")
        data = r.json()
        return data.get("message", {}).get("content", ""), data.get("prompt_eval_count")

    async def _ollama_plain(self, host: str, body: dict[str, Any]) -> tuple[str, int | None]:
        """Retry for Ollama versions without the `think` option."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{host}/api/chat", json=body)
        if r.status_code >= 400:
            raise LLMUnavailable(f"Ollama returned HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        return data.get("message", {}).get("content", ""), data.get("prompt_eval_count")

    async def _vllm(self, model: str, messages: list[dict[str, str]], schema: type[BaseModel],
                    max_tokens: int) -> tuple[str, int | None]:
        """One call to an OpenAI-compatible server with vLLM guided JSON."""
        from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, NotFoundError

        client = AsyncOpenAI(base_url=self._url(model), api_key="local", timeout=self.timeout)
        try:
            resp = await client.chat.completions.create(
                model=model, messages=messages, temperature=0, top_p=1, seed=42,  # type: ignore[arg-type]
                max_tokens=max_tokens, extra_body={"guided_json": schema.model_json_schema()},
            )
        except APIConnectionError as e:
            raise LLMUnavailable(f"Can't reach the LLM server at {self._url(model)}.") from e
        except APITimeoutError as e:
            raise LLMUnavailable(f"{model} took longer than {self.timeout} s.") from e
        except NotFoundError as e:
            raise LLMUnavailable(f"Model '{model}' is not served at {self._url(model)}.") from e
        usage = getattr(resp, "usage", None)
        return resp.choices[0].message.content or "", getattr(usage, "prompt_tokens", None)

    async def json_call(
        self, model: str, system: str, user: str, schema: type[T], max_retries: int, max_tokens: int = 4096,
        job_id: str = "-", on_retry: OnRetry | None = None,
    ) -> T:
        """Call the model; on invalid output retry with the error appended, up to `max_retries` times."""
        if is_thinking_model(model):
            user = user + "\n\n/no_think"
        messages: list[dict[str, str]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        num_ctx = context_size(system, user, max_tokens, self.max_context)
        if num_ctx == self.max_context and estimate_tokens(system + user) + max_tokens > self.max_context:
            log.warning("prompt may exceed the %d-token context of %s", self.max_context, model, extra={"job_id": job_id})
        last_error = ""
        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()
            if self.backend == "vllm":
                text, prompt_tokens = await self._vllm(model, messages, schema, max_tokens)
            else:
                text, prompt_tokens = await self._ollama(model, messages, schema, max_tokens, num_ctx)
            log.info(
                "llm %s attempt=%d prompt_tokens=%s ctx=%d latency=%.1fs",
                model, attempt, prompt_tokens or estimate_tokens(system + user), num_ctx, time.perf_counter() - t0,
                extra={"job_id": job_id},
            )
            try:
                return parse_output(text, schema)
            except InvalidModelOutput as e:
                last_error = str(e)
                log.warning("invalid %s output: %s", schema.__name__, last_error, extra={"job_id": job_id})
                if attempt < max_retries and on_retry:
                    await on_retry(f"model returned invalid JSON, retrying ({attempt + 2} of {max_retries + 1})")
                messages += [
                    {"role": "assistant", "content": text[:4000]},
                    {"role": "user", "content": REPAIR_MESSAGE.format(error=last_error)},
                ]
        raise InvalidModelOutput(f"{model} returned invalid JSON {max_retries + 1} times: {last_error}")


async def ollama_status(host: str, models: list[str], timeout: float = 3.0) -> dict[str, Any]:
    """Whether Ollama is reachable and which of `models` are downloaded (for /api/health)."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{host}/api/tags")
        names = {m.get("name", "") for m in r.json().get("models", [])}
    except Exception:  # noqa: BLE001
        return {"reachable": False, "missing": models}
    have = names | {n.removesuffix(":latest") for n in names}
    return {"reachable": True, "missing": [m for m in models if m not in have and f"{m}:latest" not in names]}
