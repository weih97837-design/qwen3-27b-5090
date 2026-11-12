"""The native Windows package (scripts\\windows\\package.ps1, docs/deploy-windows.md): where the weights and q27.dll are,
as start.cmd and the tray app set them, for the scripts that run from the package's app\\scripts (tiered_cache_test.py,
tiered_cache_bench.py). Anywhere else (the repository, the dev container) nothing changes: the variables are read as before.

Import this before q27lib and qwen27b: both read HF_HOME and Q27_LIB when they are imported.
"""

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent  # the repository, or the package's app folder


def apply():
    """Sets Q27_LIB (the engine's q27.dll, unless set) and, in the package, HF_HOME as the launcher does: settings.ini,
    else <package>\\huggingface (start_package.server_env). Returns the weight profile the server starts with in the
    package (WEIGHTS, else gdnfp4), and nvidia elsewhere (the scripts' default)."""
    dll = ROOT / "engine" / "build" / "q27.dll"
    if "Q27_LIB" not in os.environ and dll.is_file():
        os.environ["Q27_LIB"] = str(dll)
    pkg = ROOT.parent
    if ROOT.name != "app" or not (pkg / "start.cmd").is_file():
        return "nvidia"
    # The package's embeddable Python does not put this folder on sys.path: start_package.py sits beside this file.
    sys.path.insert(0, str(HERE))
    import start_package  # the launcher's own settings reader (the same rules as start.cmd and the tray app)

    env, _ = start_package.server_env(os.environ, start_package.read_settings(pkg / "settings.ini"), pkg)
    os.environ["HF_HOME"] = env["HF_HOME"]
    return env.get("WEIGHTS") or "gdnfp4"
