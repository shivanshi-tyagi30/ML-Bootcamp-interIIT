"""Parse model output as JSON and validate it against a Pydantic schema."""

from __future__ import annotations

import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

REPAIR_MESSAGE = "Your previous output was invalid: {error}. Return ONLY corrected JSON matching the schema."


class InvalidModelOutput(Exception):
    """The model output could not be parsed or validated."""


def extract_json(text: str) -> str:
    """Strip markdown fences, <think> blocks and any text before the first '{' / after the last '}'."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise InvalidModelOutput("no JSON object found")
    return text[start : end + 1]


def parse_output(text: str, schema: type[T]) -> T:
    """Validate model text as `schema`; raises InvalidModelOutput with a short reason."""
    try:
        return schema.model_validate_json(extract_json(text))
    except ValidationError as e:
        errs = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:5])
        raise InvalidModelOutput(errs) from e
