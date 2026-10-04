"""Render README.md and docs/methodology.md from their templates and ``docs/results.json``.

Templates live in ``docs/templates``. A placeholder ``{{ path/to/value | format }}`` is
replaced by a value of the results file, so every number in the text is read from the
results rather than typed. Formats: an integer ``N`` (``N`` decimals), ``pctN`` (percent with
``N`` decimals), ``int`` (thousands separators), ``str``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "docs" / "templates"
TARGETS = {
    "README.template.md": ROOT / "README.md",
    "methodology.template.md": ROOT / "docs" / "methodology.md",
}
PLACEHOLDER = re.compile(r"\{\{\s*([^|}]+?)\s*(?:\|\s*([^}]+?)\s*)?\}\}")


def lookup(data, path: str):
    node = data
    for part in path.split("/"):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def fmt(value, spec: str | None) -> str:
    if value is None:
        raise ValueError("value is null")
    spec = spec or "str"
    if spec == "str":
        return str(value)
    if spec == "int":
        return f"{round(value):,}"
    if spec.startswith("pct"):
        return f"{100 * value:.{int(spec[3:] or 0)}f}%"
    if spec.isdigit():
        return f"{value:.{int(spec)}f}"
    raise ValueError(f"unknown format {spec!r}")


def render(text: str, results: dict) -> str:
    def sub(match: re.Match) -> str:
        path, spec = match.group(1), match.group(2)
        try:
            return fmt(lookup(results, path), spec)
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise ValueError(f"cannot resolve placeholder {match.group(0)!r}: {exc!r}") from exc

    return PLACEHOLDER.sub(sub, text)


def main() -> None:
    results = json.loads((ROOT / "docs" / "results.json").read_text())
    for name, target in TARGETS.items():
        target.write_text(render((TEMPLATES / name).read_text(), results), encoding="utf-8")
        print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
