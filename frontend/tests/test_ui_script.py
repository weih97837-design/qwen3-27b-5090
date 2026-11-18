"""The status page's inline script must at least parse: a stray `: (...)` in a chained ternary (fmtVal, once) made the whole
script fail to load and left the page blank, and no Python test noticed."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1] / "ui" / "index.html"


def inline_scripts():
    return re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", UI.read_text(encoding="utf-8"), flags=re.S)


def test_the_page_has_one_inline_script():
    assert len(inline_scripts()) == 1  # (the check below covers every script; a second one would need no change)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_inline_script_parses(tmp_path):
    for i, code in enumerate(inline_scripts()):
        f = tmp_path / f"ui_{i}.js"
        f.write_text(code, encoding="utf-8")
        r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr[:600]
