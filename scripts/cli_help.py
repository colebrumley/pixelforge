"""Record the `--help` texts of the Python (click) and Rust (clap) CLIs side by side.

  python scripts/cli_help.py [--rust PATH] [--out DIR]

Writes DIR/python-<cmd>.txt and DIR/rust-<cmd>.txt for the group (`<cmd>` = ``group``) and
each subcommand; DIR defaults to tests/cli_help/. The Rust binary is ``--rust``, else
``$PIXELFORGE_BIN``, else target/release/pixelforge. The click help is rendered in process
as ``pixelforge`` at 80 columns, so the records do not depend on the terminal.

The layouts differ (click aligns the help in a column and wraps at 80 columns, shows both
halves of an on/off pair and choices in brackets; clap puts long help on the next line,
does not wrap, writes values in angle brackets and lists the positional arguments), so
tests/test_cli_parity.py compares what matters with ``options``: the option names and
their ``[default: ...]`` texts.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMANDS = ("group", "batch", "compare", "convert", "palettes")
FLAG = re.compile(r"--[A-Za-z0-9][A-Za-z0-9-]*")
DEFAULT = re.compile(r"\[default: ([^\]]*)\]")


def _args(command: str) -> list[str]:
    return ["--help"] if command == "group" else [command, "--help"]


def python_help(command: str) -> str:
    """The click CLI's help for `command` (``group`` for the group itself)."""
    from click.testing import CliRunner

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from pixelforge.cli import cli

    result = CliRunner().invoke(cli, _args(command), prog_name="pixelforge",
                                terminal_width=80)
    assert result.exit_code == 0, result.output
    return result.output


def rust_help(binary: list[str], command: str) -> str:
    """The Rust program's help for `command`."""
    done = subprocess.run(binary + _args(command), capture_output=True, text=True,
                          check=True, env={k: v for k, v in os.environ.items()
                                           if k != "PIXELFORGE_DEBUG"})
    return done.stdout


def options(text: str) -> dict[str, str | None]:
    """{option name: its ``[default: ...]`` text, or None} of a help text's Options section.

    An entry starts on a line indented by at most six spaces whose text starts with ``-``;
    its declaration is the text before the first run of two spaces, its help the rest plus
    the more indented lines that follow (whitespace collapsed, so wrapping does not matter).
    Every name of a declaration (``-o, --outdir``, ``--key-bg / --no-key-bg``) maps to the
    entry's default.
    """
    entries: list[tuple[str, list[str]]] = []
    in_options = False
    for line in text.splitlines():
        if line.startswith("Options:"):
            in_options = True
            continue
        if not in_options:
            continue
        if line and not line[0].isspace():
            break                       # the next section, or clap's after-help
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith("-") and indent <= 6:
            declaration, *help_text = re.split(r"\s{2,}", stripped, maxsplit=1)
            entries.append((declaration, help_text))
        elif entries and stripped:
            entries[-1][1].append(stripped)
    found: dict[str, str | None] = {}
    for declaration, help_lines in entries:
        help_text = " ".join(" ".join(help_lines).split())
        match = DEFAULT.search(help_text)
        for name in FLAG.findall(declaration):
            found[name] = match.group(1) if match else None
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rust", default=os.environ.get("PIXELFORGE_BIN")
                        or str(ROOT / "target" / "release" / "pixelforge"))
    parser.add_argument("--out", type=Path, default=ROOT / "tests" / "cli_help")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for command in COMMANDS:
        (args.out / f"python-{command}.txt").write_text(python_help(command))
        (args.out / f"rust-{command}.txt").write_text(rust_help([args.rust], command))
        print(f"{command}: {args.out}/{{python,rust}}-{command}.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
