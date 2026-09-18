"""Compatibility shim — makes ``org_rag_phase1.*`` resolve to *this* copy.

Every module in this project imports itself as ``org_rag_phase1.<module>``
(``org_rag_phase1.config``, ``org_rag_phase1.src.agents.recon``, ...), which is
the package name the project had when the repository root was the package
root. This folder is no longer named that way, and a *sibling* checkout with
that name may exist next to it — in which case plain ``sys.path`` resolution
used to import the wrong copy (or fail outright).

Rather than rewriting ~90 import statements, this package redirects its own
search path to the project root, so:

    org_rag_phase1.config              -> <project>/config.py
    org_rag_phase1.src.agents.recon    -> <project>/src/agents/recon.py

Import order matters: callers must put the project root on ``sys.path`` *before*
any parent directory that also contains an ``org_rag_phase1`` entry (see the
bootstrap block at the top of ``scripts/*.py`` and ``tests/conftest.py``).
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Mirrors a package's __path__ so submodule lookups land on the project root.
__path__ = [str(_PROJECT_ROOT)]

__all__: list[str] = []
