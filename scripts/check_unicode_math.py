#!/usr/bin/env python3
"""Reject Unicode super/subscript characters in executable note sources."""

from __future__ import annotations

import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = ("chapters", "appendices", "code", "scripts")
SOURCE_SUFFIXES = {".qmd", ".py"}
FORBIDDEN_NAME_PARTS = (
    "SUPERSCRIPT",
    "SUBSCRIPT",
    "MODIFIER LETTER SMALL",
)


def source_paths() -> list[Path]:
    paths = set(PROJECT_ROOT.glob("*.qmd"))
    for directory_name in SOURCE_DIRECTORIES:
        directory = PROJECT_ROOT / directory_name
        if directory.is_dir():
            paths.update(
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix in SOURCE_SUFFIXES
            )
    return sorted(paths)


def forbidden_characters(line: str) -> list[tuple[int, str, str]]:
    matches = []
    for column, character in enumerate(line, start=1):
        unicode_name = unicodedata.name(character, "")
        if any(part in unicode_name for part in FORBIDDEN_NAME_PARTS):
            matches.append((column, character, unicode_name))
    return matches


def main() -> int:
    violations = []
    for path in source_paths():
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for column, character, unicode_name in forbidden_characters(line):
                violations.append(
                    (
                        path.relative_to(PROJECT_ROOT),
                        line_number,
                        column,
                        character,
                        unicode_name,
                    )
                )

    if not violations:
        print("Unicode 数学字符检查通过。")
        return 0

    print("检测到 Unicode 数学上下标字符：")
    for path, line, column, character, unicode_name in violations:
        print(
            f"  {path}:{line}:{column}: {character!r} "
            f"(U+{ord(character):04X} {unicode_name})"
        )
    print("请改用 LaTeX/MathText 语法，例如 `$x^2$`、`$x_0$` 或 `r\"$x^2$\"`。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
