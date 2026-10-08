"""Generated TeX must contain no Python escape-sequence corruption.

Writing LaTeX from a NON-raw Python string lets the parser expand the escape
before the text reaches the file:

    "\\varphi"  -> VT  + "arphi"     (0x0b)
    "\\texttt"  -> TAB + "exttt"     (0x09)
    "\\times"   -> TAB + "imes"      (0x09)
    "\\rho"     -> CR  + "ho"        (0x0d)
    "\\f..."    -> FF  + ...         (0x0c)

The result still looks like plausible LaTeX in an editor but fails to
compile, and the failure message points at \\cmidrule or \\noalign rather
than at the real cause. This happened to tab_dependency.tex and
tab_optlevel.tex and is the reason every table generator now uses raw
strings.

CR is tolerated only as part of a CRLF line ending, which is what a Windows
checkout produces and which LaTeX reads fine.

Run: python tests/test_generated_tex.py   (exit 0 on pass)
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX_DIRS = [os.path.join(ROOT, "paper", "paper_mdpi", "tables")]
TEX_FILES = [os.path.join(ROOT, "paper", "paper_mdpi", "main.tex"),
             os.path.join(ROOT, "paper", "paper_mdpi", "supplementary.tex")]

# the mangled remnants a swallowed backslash leaves behind
ORPHANS = ("arphi", "exttt", "imes", "ho ", "rac", "ootnotesize", "extbf")


def targets():
    out = list(TEX_FILES)
    for d in TEX_DIRS:
        out += sorted(glob.glob(os.path.join(d, "*.tex")))
    return [p for p in out if os.path.exists(p)]


def test_no_control_characters():
    bad = []
    for path in targets():
        raw = open(path, "rb").read()
        for i, byte in enumerate(raw):
            if byte in (0x0b, 0x0c):                    # VT, FF: never valid
                bad.append((path, i, byte))
            elif byte == 0x09:                          # TAB
                # a tab is fine as leading indentation, never mid-command
                line_start = raw.rfind(b"\n", 0, i) + 1
                if raw[line_start:i].strip():
                    bad.append((path, i, byte))
            elif byte == 0x0d:                          # CR
                if raw[i + 1:i + 2] != b"\n":           # not a CRLF ending
                    bad.append((path, i, byte))
    for path, i, byte in bad[:10]:
        raw = open(path, "rb").read()
        ctx = raw[max(0, i - 28):i + 12].decode("utf-8", "replace")
        print("    0x%02x in %s: ...%s..."
              % (byte, os.path.basename(path), ctx.replace("\n", "\\n")))
    assert not bad, "%d stray control character(s) in generated TeX" % len(bad)
    print("  no stray control characters in %d TeX files" % len(targets()))


def test_no_orphaned_commands():
    """A swallowed backslash leaves a recognisable word fragment behind."""
    bad = []
    for path in targets():
        text = open(path, encoding="utf-8", errors="replace").read()
        for orphan in ORPHANS:
            for m in re.finditer(r"(?<![A-Za-z\\])" + orphan, text):
                # legitimate when preceded by a backslash, i.e. a real command
                if text[max(0, m.start() - 1):m.start()] != "\\":
                    bad.append((os.path.basename(path), orphan,
                                text[max(0, m.start() - 24):m.start() + 14]))
    for name, orphan, ctx in bad[:10]:
        print("    %-24s orphan %-12s ...%s..."
              % (name, orphan, ctx.replace("\n", "\\n")))
    assert not bad, "%d orphaned LaTeX command fragment(s)" % len(bad)
    print("  no orphaned command fragments")


def test_table_rows_end_with_double_backslash():
    """A row ending in a single backslash breaks the next \\cmidrule."""
    bad = []
    for path in targets():
        if "tables" not in path:
            continue
        for n, line in enumerate(
                open(path, encoding="utf-8").read().split("\n"), 1):
            s = line.rstrip()
            if not s or s.lstrip().startswith("%"):
                continue
            # a line ending in exactly one backslash is almost always a
            # mangled row terminator
            trailing = len(s) - len(s.rstrip("\\"))
            if trailing == 1:
                bad.append((os.path.basename(path), n, s[-60:]))
    for name, n, s in bad[:10]:
        print("    %s line %d: ...%s" % (name, n, s))
    assert not bad, "%d table row(s) ending in a single backslash" % len(bad)
    print("  every table row terminator is a double backslash")


def test_balanced_braces():
    for path in targets():
        text = open(path, encoding="utf-8").read()
        text = re.sub(r"(?<!\\)%.*", "", text)
        depth = 0
        for i, ch in enumerate(text):
            if ch in "{}" and (i == 0 or text[i - 1] != "\\"):
                depth += 1 if ch == "{" else -1
                assert depth >= 0, "unbalanced brace in %s" % path
        assert depth == 0, "%d unclosed brace(s) in %s" % (depth, path)
    print("  braces balanced in every TeX file")


if __name__ == "__main__":
    test_no_control_characters()
    test_no_orphaned_commands()
    test_table_rows_end_with_double_backslash()
    test_balanced_braces()
    print("\ngenerated-TeX integrity tests passed")
