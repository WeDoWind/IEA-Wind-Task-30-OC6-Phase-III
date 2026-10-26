"""Read whitespace-delimited numeric text files exactly, and account for every value.

Both source archives are plain text: one or more header lines, then a block of numbers.
:func:`read_numeric` returns the header verbatim and the numbers as float64, together with
the number of tokens the block held, so that a converter can prove it stored every value
somewhere rather than assume it.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Tokens that stand for a missing value in the participants' files.
NA_TOKENS = ("NaN", "nan", "NAN", "Nan", "-", "***")


@dataclass
class NumericText:
    """One text file split into its header and its numeric block."""

    header: list[str]
    values: np.ndarray
    n_tokens: int
    encoding: str
    notes: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return self.values.shape[0]

    @property
    def n_columns(self) -> int:
        return self.values.shape[1] if self.values.ndim == 2 else 0


def decode(raw: bytes) -> tuple[str, str]:
    """UTF-8 as a rule; UTF-16 for the few files saved that way (BOM or NUL bytes)."""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or b"\x00" in raw[:64]:
        return raw.decode("utf-16", errors="replace"), "utf-16"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


def is_numeric_row(line: str) -> bool:
    tokens = line.replace(",", " ").split()
    if not tokens:
        return False
    try:
        [float(token) for token in tokens if token not in NA_TOKENS]
    except ValueError:
        return False
    return True


def read_numeric(raw: bytes, fixes: tuple[str, str] | None = None) -> NumericText:
    """Split ``raw`` into header lines and a float64 block.

    The header is every line before the first one that parses as numbers - detected by
    parsing rather than by looking for letters, because one participant labels its columns
    ``05:X 07:Z ...``. ``fixes`` is an ``(old, new)`` literal repair applied to the text
    first. Rows of unequal length are padded with NaN and reported in ``notes``; the
    padding is not counted in ``n_tokens``.
    """
    text, encoding = decode(raw)
    notes = []
    if fixes is not None and fixes[0] in text:
        text = text.replace(*fixes)
        notes.append(f"repaired {fixes[0]!r} -> {fixes[1]!r}")
    lines = text.splitlines()
    start = 0
    while start < len(lines) and not is_numeric_row(lines[start]):
        start += 1
    header = [line.rstrip() for line in lines[:start] if line.strip()]
    body = [line for line in lines[start:] if line.strip()]
    if any("," in line for line in body):
        body = [line.replace(",", " ") for line in body]
        notes.append("comma separated")
    n_tokens = sum(len(line.split()) for line in body)
    if not body:
        return NumericText(header, np.zeros((0, 0)), 0, encoding, notes)

    widths = {len(line.split()) for line in body}
    if len(widths) == 1:
        values = pd.read_csv(
            io.StringIO("\n".join(body)), sep=r"\s+", header=None, engine="c",
            dtype=np.float64, na_values=list(NA_TOKENS), keep_default_na=False,
        ).to_numpy()
    else:
        width = max(widths)
        values = np.full((len(body), width), np.nan)
        for i, line in enumerate(body):
            tokens = line.split()
            values[i, :len(tokens)] = [
                np.nan if token in NA_TOKENS else float(token) for token in tokens
            ]
        notes.append(f"ragged rows ({min(widths)}-{width} values), padded with NaN")
    return NumericText(header, values, n_tokens, encoding, notes)
