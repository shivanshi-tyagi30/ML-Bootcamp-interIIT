"""LLM client for Ollama (native API) or vLLM (OpenAI-compatible) with schema-constrained JSON (spec Section 12).

Ollama is called through its native /api/chat endpoint rather than the OpenAI-compatible one, because only
the native API lets us set the context window (Ollama's small default silently truncates long prompts),
switch off Qwen3's "thinking" (very slow on CPU) and keep the model loaded between calls.
"""

from __future__ import annotations

import asyncio
import logging
import re
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
CONTEXT_BUCKETS = (8192, 12288, 16384, 32768, 65536, 131072)


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

    async def preload(self, model: str) -> bool:
        """Load a model into Ollama's memory ahead of the first job; False if it could not be loaded."""
        if self.backend != "ollama":
            return True
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(f"{self._url(model).removesuffix('/v1')}/api/generate",
                                  json={"model": model, "keep_alive": self.keep_alive,
                                        # same context as a short meeting's calls, so Ollama doesn't reload
                                        "options": {"num_ctx": min(CONTEXT_BUCKETS[0], self.max_context)}})
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}")
            log.info("preloaded %s", model)
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("could not preload %s (%r)", model, e)
            return False

    async def unload(self, model: str) -> None:
        """Ask Ollama to free a model's memory now (best effort; frees RAM for the next model on a laptop)."""
        if self.backend != "ollama":
            return
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                await client.post(f"{self._url(model).removesuffix('/v1')}/api/generate",
                                  json={"model": model, "keep_alive": 0})
        except Exception:  # noqa: BLE001
            log.info("could not unload %s", model)

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


GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
CLOUD_MIN_OUTPUT_TOKENS = 8192  # thinking models spend part of the output budget on reasoning


def inline_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema with $refs inlined and titles/defaults removed: the subset cloud APIs accept."""
    full = schema.model_json_schema()
    defs = full.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k not in ("title", "default")}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(full)


RESOLVED_MODELS: dict[tuple[str, str], str] = {}  # (base_url, requested model) -> model actually available


class GeminiClient:
    """Cloud LLM through an OpenAI-compatible chat API (Google Gemini by default; GPT, Groq etc. by URL).

    - Asks for our exact JSON structure (`response_format: json_schema`). A provider that rejects it gets
      plain JSON mode with the structure written into the prompt instead.
    - Leaves room for "thinking" models: the output budget is never below CLOUD_MIN_OUTPUT_TOKENS.
    - Clear errors for a bad key, rate limits and timeouts; never a silent switch to a slower model.
    - With a key the cloud model is the only model: no internet is a clear error, never the slow local model.
    """

    def __init__(
        self, api_key: str, model: str = "gemini-2.5-flash", timeout: int = 300,
        base_url: str = GEMINI_OPENAI_URL, reasoning_effort: str = "low",
    ) -> None:
        """Create a client for one cloud model (used for both LM1 and LM2)."""
        self.api_key = api_key.strip()
        self.requested = model.strip() or "gemini-2.5-flash"
        self.base_url = base_url.rstrip("/")
        # A retired model replaced once is replaced for every later job too (no repeated 404s).
        self.model = RESOLVED_MODELS.get((self.base_url, self.requested), self.requested)
        # A model that was busy a few minutes ago: start this job on the one that answered instead.
        switched, until = BUSY_SWITCH.get((self.base_url, self.requested), ("", 0.0))
        if switched and time.monotonic() < until:
            self.model = switched
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort.strip()
        self.schema_in_prompt = False  # set after a provider rejects json_schema
        self._available: list[str] = []
        self._tried: set[str] = set()

    @property
    def name(self) -> str:
        """Model label for the record."""
        return f"{self.model} (cloud)"

    async def preload(self, model: str) -> bool:
        """Nothing to load for a cloud model."""
        return True

    async def unload(self, model: str) -> None:
        """Nothing to unload for a cloud model."""

    def _body(self, messages: list[dict[str, str]], schema: type[BaseModel], max_tokens: int) -> dict[str, Any]:
        """Request body for one call."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max(max_tokens * 2, CLOUD_MIN_OUTPUT_TOKENS),
        }
        if self.schema_in_prompt:
            body["response_format"] = {"type": "json_object"}
        else:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": inline_schema(schema), "strict": False}}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def _timeout_for(self, body: dict[str, Any]) -> float:
        """Seconds to wait for one answer: enough for the input size, so a stalled request is given up on
        (and another model tried) in about a minute rather than after CLOUD_TIMEOUT_SEC."""
        chars = sum(len(m.get("content") or "") for m in body["messages"])
        return min(float(self.timeout), CLOUD_MIN_TIMEOUT + chars / 1000)

    async def _post(self, body: dict[str, Any]) -> httpx.Response:
        """POST, with one short wait-and-retry on rate limits and server errors ("high demand"); after that the
        caller tries another model, which is much faster than waiting for a busy one."""
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(CLOUD_BUSY_RETRIES + 1):
            async with httpx.AsyncClient(timeout=self._timeout_for(body)) as client:
                r = await client.post(url, headers=headers, json=body)
            if r.status_code in BUSY and attempt < CLOUD_BUSY_RETRIES:
                wait = min(float(r.headers.get("retry-after") or 0) or CLOUD_BUSY_WAIT, CLOUD_MAX_WAIT)
                log.warning("cloud LLM HTTP %d; retrying in %.0f s", r.status_code, wait)
                await asyncio.sleep(wait)
                continue
            return r
        return r

    async def _discover_model(self) -> str | None:
        """Best available chat model for this key (newest "flash" model, else newest "pro"), or None."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.api_key}"})
            ids = [str(m.get("id", "")).removeprefix("models/") for m in r.json().get("data", [])]
        except Exception as e:  # noqa: BLE001
            log.warning("could not list cloud models (%r)", e)
            return None
        self._available = ids
        return pick_chat_model(ids)

    async def _busy_backup(self) -> str | None:
        """Another chat model for this key, for when the current one stays overloaded (each has its own capacity
        and free-tier quota). Tried models are not picked again within this job."""
        self._tried.add(self.model)
        if not self._available:
            await self._discover_model()
        return pick_chat_model([m for m in self._available if m not in self._tried])

    async def _switch(self, why: str, notify: OnRetry | None) -> bool:
        """Move to another model after the current one was busy or too slow; False when none is left."""
        backup = await self._busy_backup()
        if not backup:
            return False
        log.warning("cloud model %s %s; switching to %s", self.model, why, backup)
        if notify:
            await notify(f"{self.model} {why}, switching to {backup}")
        BUSY_SWITCH[(self.base_url, self.requested)] = (backup, time.monotonic() + BUSY_SWITCH_SEC)
        self.model = backup
        return True

    async def _call(self, messages: list[dict[str, str]], schema: type[BaseModel], max_tokens: int,
                    notify: OnRetry | None = None) -> str:
        """One completed call; adapts to providers that reject json_schema or reasoning_effort, to retired
        models, and to a model that stays overloaded (switches to another cloud model, never to the local one)."""
        for _ in range(8):
            body = self._body(messages if not self.schema_in_prompt else _with_schema(messages, schema),
                              schema, max_tokens)
            try:
                r = await self._post(body)
            except httpx.TimeoutException as e:
                if await self._switch(f"did not answer within {self._timeout_for(body):.0f} s", notify):
                    continue
                raise LLMUnavailable(f"{self.model} did not answer in time. Wait a minute and press Retry: "
                                     "finished steps are kept.") from e
            if r.status_code == 400:
                text = r.text.lower()
                if "reasoning" in text and self.reasoning_effort:
                    self.reasoning_effort = ""  # provider without that option
                    continue
                if not self.schema_in_prompt and ("schema" in text or "response_format" in text):
                    self.schema_in_prompt = True
                    continue
            if r.status_code in (401, 403):
                raise LLMUnavailable("The cloud model rejected the API key. Check the key and try again.")
            if r.status_code in BUSY and await self._switch(f"is busy (HTTP {r.status_code})", notify):
                continue
            if r.status_code == 429:
                raise LLMUnavailable("The cloud model's free-tier rate limit was reached. Wait a minute and retry.")
            if r.status_code in BUSY:
                raise LLMUnavailable("The cloud model is overloaded right now (Google says this is usually "
                                     "temporary). Wait a minute and press Retry: finished steps are kept.")
            if r.status_code == 404:
                # Model retired or renamed: ask the provider which models this key can use and pick one.
                replacement = await self._discover_model()
                if replacement and replacement != self.model:
                    log.warning("cloud model %s not found; using %s instead", self.model, replacement)
                    RESOLVED_MODELS[(self.base_url, self.requested)] = replacement
                    self.model = replacement
                    continue
                avail = ", ".join(self._available[:12]) or "none listed"
                raise LLMUnavailable(f"Model '{self.model}' was not found. Models this key can use: {avail}. "
                                     "Set GEMINI_MODEL to one of them.")
            if r.status_code >= 400:
                raise LLMUnavailable(f"Cloud model returned HTTP {r.status_code}: {r.text[:300]}")
            choice = (r.json().get("choices") or [{}])[0]
            if choice.get("finish_reason") == "length":
                log.warning("cloud LLM answer was cut off at max_tokens=%s", body["max_tokens"])
            return (choice.get("message") or {}).get("content") or ""
        raise LLMUnavailable("The cloud model rejected the request format.")

    async def json_call(
        self, model: str, system: str, user: str, schema: type[T], max_retries: int, max_tokens: int = 4096,
        job_id: str = "-", on_retry: OnRetry | None = None,
    ) -> T:
        """Return a validated instance of `schema` from the cloud model."""
        messages: list[dict[str, str]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error = ""
        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()
            try:
                text = await self._call(messages, schema, max_tokens, on_retry)
            except httpx.TransportError as e:  # a key means cloud only: never a silent switch to the local model
                raise LLMUnavailable("Can't reach the cloud model. Check the internet connection and press Retry: "
                                     "finished steps are kept.") from e
            log.info("cloud llm %s attempt=%d schema=%s latency=%.1fs", self.model, attempt, schema.__name__,
                     time.perf_counter() - t0, extra={"job_id": job_id})
            try:
                return parse_output(text, schema)
            except InvalidModelOutput as e:
                last_error = str(e)
                log.warning("invalid %s output from %s: %s", schema.__name__, self.model, last_error,
                            extra={"job_id": job_id})
                if attempt < max_retries and on_retry:
                    await on_retry(f"model returned invalid JSON, retrying ({attempt + 2} of {max_retries + 1})")
                messages = messages + [
                    {"role": "assistant", "content": text[:4000]},
                    {"role": "user", "content": REPAIR_MESSAGE.format(error=last_error)},
                ]
        raise InvalidModelOutput(f"{self.model} returned invalid JSON {max_retries + 1} times: {last_error}")


BUSY = (429, 500, 502, 503, 504)
CLOUD_BUSY_RETRIES = 1  # one short wait, then another model (a busy model often stays busy for minutes)
CLOUD_BUSY_WAIT = 2.0
CLOUD_MAX_WAIT = 10.0  # cap on the provider's retry-after
CLOUD_MIN_TIMEOUT = 60.0  # seconds for a short request; +1 s per 1000 characters of input
BUSY_SWITCH: dict[tuple[str, str], tuple[str, float]] = {}  # (url, requested) -> (model that answered, until)
BUSY_SWITCH_SEC = 15 * 60

NOT_CHAT = ("embed", "image", "tts", "audio", "live", "vision", "aqa", "veo", "imagen", "learnlm", "robotics",
            "computer-use", "native", "transcribe", "lyria", "nano", "gemma")


def pick_chat_model(ids: list[str]) -> str | None:
    """Newest general chat model: prefer "flash" (fast), skip "lite", previews/experiments if a stable one exists."""
    def version(m: str) -> tuple[float, ...]:
        nums = re.findall(r"(\d+(?:\.\d+)?)", m)
        return tuple(float(n) for n in nums[:2]) or (0.0,)

    chat = [m for m in ids if m.startswith("gemini") and not any(x in m for x in NOT_CHAT)] or \
           [m for m in ids if not any(x in m for x in NOT_CHAT)]
    for want in ("flash", "pro", ""):
        pool = [m for m in chat if want in m and "lite" not in m] or [m for m in chat if want in m]
        stable = [m for m in pool if not re.search(r"preview|exp|latest|\d{2}-\d{2}", m)] or pool
        if stable:
            return max(stable, key=lambda m: (version(m), -len(m)))
    return None


def _with_schema(messages: list[dict[str, str]], schema: type[BaseModel]) -> list[dict[str, str]]:
    """Messages with the JSON structure appended to the last user turn (for plain JSON mode)."""
    import json

    out = [dict(m) for m in messages]
    out[-1]["content"] += "\n\nReturn JSON with exactly this structure (JSON Schema):\n" + json.dumps(
        inline_schema(schema), ensure_ascii=False)
    return out
