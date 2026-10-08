"""Markdown record export via Jinja2 (spec Section 10.14)."""

from __future__ import annotations

from jinja2 import Environment, StrictUndefined

from app.exports.txt import hms
from app.models.record import MeetingRecord

TEMPLATE = """\
# {{ r.meta.title }}

- **Date:** {{ date(r.meta.generated_at) }}
- **Duration:** {{ hms(r.meta.duration_s) }}
- **Source file:** {{ r.meta.source_file }}
{% if r.meta.warnings %}- **Warnings:** {{ r.meta.warnings | join(", ") }}
{% endif %}
## Summary

{% for s in r.summary %}{{ s.text }}

{% else %}No summary was produced.

{% endfor %}## Minutes

{% for t in r.minutes %}### {{ t.topic }}

{% for p in t.points %}- {{ p.text }}
{% endfor %}
{% else %}No minutes were produced.

{% endfor %}## Key decisions

{% for d in r.decisions %}- **{{ d.id }}** {{ d.decision }} (agreement: "{{ d.agreement_evidence }}")
{% else %}No decisions were reached.
{% endfor %}
## Not agreed (open proposals)

{% for p in r.open_proposals %}- {{ p.proposal }}{% if p.demoted_from_decision %} _(no clear agreement)_{% endif %}
{% else %}None.
{% endfor %}
## Action items

{% if r.action_items %}| # | Task | Owner | Deadline |
|---|---|---|---|
{% for a in r.action_items %}| {{ a.id }} | {{ cell(a.task) }} | {{ cell(a.owner) }} | {{ cell(a.deadline) }} |
{% endfor %}{% else %}No action items were identified.
{% endif %}
"""

_env = Environment(undefined=StrictUndefined, keep_trailing_newline=True, autoescape=False)
_env.globals.update(
    hms=hms,
    date=lambda d: f"{d.day} {d:%b %Y}",
    cite=lambda ids: "",
    cell=lambda s: str(s).replace("|", "\\|").replace("\n", " "),
)
_template = _env.from_string(TEMPLATE)


def record_markdown(record: MeetingRecord) -> str:
    """Render the record as Markdown."""
    return _template.render(r=record)
