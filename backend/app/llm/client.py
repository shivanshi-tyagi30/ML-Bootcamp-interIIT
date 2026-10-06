"""OpenAI-compatible client for vLLM or Ollama with schema-constrained JSON (spec Section 12)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel

from app.core.text import estimate_tokens
from app.llm.json_repair import REPAIR_MESSAGE, InvalidModelOutput, parse_output

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Read a prompt file from app/llm/prompts/."""
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


class JSONLLM(Protocol):
    """What the pipeline needs from an LLM; tests provide a fake."""

    async def json_call(
        self, model: str, system: str, user: str, schema: type[T], max_retries: int, max_tokens: int = 4096,
        job_id: str = "-",
    ) -> T:
        """Return a validated instance of `schema`."""
        ...


class LLMClient:
    """Calls the model at temperature 0 with JSON-schema constrained output and validates it."""

    def __init__(
        self, base_url: str, backend: Literal["ollama", "vllm"], timeout: int, model_urls: dict[str, str] | None = None,
    ) -> None:
        """Create a client for an OpenAI-compatible server; `model_urls` routes specific models elsewhere."""
        from openai import AsyncOpenAI

        self.backend = backend
        self.client = AsyncOpenAI(base_url=base_url, api_key="local", timeout=timeout)
        self.routes = {m: AsyncOpenAI(base_url=u, api_key="local", timeout=timeout) for m, u in (model_urls or {}).items() if u}

    def _format_args(self, schema: type[BaseModel]) -> dict:
        """Backend-specific arguments that constrain output to the schema."""
        js = schema.model_json_schema()
        if self.backend == "vllm":
            return {"extra_body": {"guided_json": js}}
        return {"response_format": {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": js}}}

    async def json_call(
        self, model: str, system: str, user: str, schema: type[T], max_retries: int, max_tokens: int = 4096,
        job_id: str = "-",
    ) -> T:
        """Call the model; on invalid output retry with the error appended, up to `max_retries` times."""
        messages: list[dict[str, str]] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error = ""
        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()
            resp = await self.routes.get(model, self.client).chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0,
                top_p=1,
                seed=42,
                max_tokens=max_tokens,
                **self._format_args(schema),
            )
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            log.info(
                "llm %s attempt=%d prompt_tokens=%s latency=%.1fs",
                model, attempt, getattr(usage, "prompt_tokens", None) or estimate_tokens(system + user),
                time.perf_counter() - t0, extra={"job_id": job_id},
            )
            try:
                return parse_output(text, schema)
            except InvalidModelOutput as e:
                last_error = str(e)
                log.warning("invalid %s output: %s", schema.__name__, last_error, extra={"job_id": job_id})
                messages += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": REPAIR_MESSAGE.format(error=last_error)},
                ]
        raise InvalidModelOutput(last_error)
