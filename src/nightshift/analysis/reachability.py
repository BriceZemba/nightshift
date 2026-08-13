"""Static reachability analysis - is the vulnerable code actually used here?

This is the judgement that makes Nightshift worth running. A dependency appearing in a
lockfile means the advisory *could* apply; it says nothing about whether the repository
ever calls the vulnerable code. Most do not. Separating those two cases is what turns 180
advisories into the handful a maintainer should actually read.

Scope, stated honestly
----------------------
This is an **import-and-usage** analysis over Python source, not interprocedural
call-graph analysis:

* It finds every import of the vulnerable package, resolving aliases and ``from`` imports.
* It finds every site where an imported name is used, and records file, line and snippet.
* When the advisory names specific vulnerable symbols, it narrows to those symbols.

What it deliberately does **not** do: follow calls through helper functions to decide
whether a code path is live at runtime, resolve dynamic dispatch, or handle
``importlib``/``__import__``. Full call-graph analysis is a research project, and claiming
it without doing it would be worse than a bounded analysis with a stated limit.

The bias throughout is toward :data:`Reachability.UNKNOWN` rather than a confident
``NOT_REACHABLE``. A false "not reachable" hides a real vulnerability; a false "unknown"
costs a human thirty seconds.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

import structlog

from nightshift.models import CallSite, Reachability

log = structlog.get_logger(__name__)

#: Modules imported dynamically defeat static analysis entirely. Their presence in a file
#: that also imports the vulnerable package downgrades a NOT_REACHABLE verdict to UNKNOWN.
_DYNAMIC_IMPORT_MARKERS = frozenset({"importlib", "__import__", "pkgutil", "pkg_resources"})


@dataclass
class ImportBinding:
    """One local name bound to something from the vulnerable package.

    ``import requests as rq`` binds ``rq`` -> ``requests``.
    ``from requests.sessions import Session`` binds ``Session`` -> ``requests.sessions.Session``.
    """

    local_name: str
    qualified_name: str
    line: int
    #: True for ``from x import y``, where the local name refers to a symbol rather than
    #: a module. Determines how a usage site is reconstructed into a qualified name.
    is_symbol: bool


@dataclass
class FileAnalysis:
    path: str
    bindings: list[ImportBinding] = field(default_factory=list)
    call_sites: list[CallSite] = field(default_factory=list)
    has_dynamic_imports: bool = False
    parse_failed: bool = False


@dataclass
class ReachabilityResult:
    reachability: Reachability
    call_paths: list[CallSite] = field(default_factory=list)
    rationale: str = ""
    files_analyzed: int = 0
    files_failed: int = 0


def _module_matches(module: str, package: str) -> bool:
    """True when ``module`` is the package itself or a submodule of it.

    Prefix matching alone is wrong: ``requests_oauthlib`` starts with ``requests`` but is a
    different distribution entirely, so the boundary must be a dot.
    """
    if not module:
        return False
    return module == package or module.startswith(f"{package}.")


def _import_names(package: str) -> set[str]:
    """Import names a distribution might plausibly use.

    PyPI distribution names and Python import names diverge often enough to matter --
    ``beautifulsoup4`` imports as ``bs4``, ``pyyaml`` as ``yaml``. The general rule
    (hyphens become underscores) covers most of the rest; the explicit table covers the
    common offenders. An unknown mapping means the analysis under-reports, so the caller
    treats "no usage found" as UNKNOWN rather than NOT_REACHABLE when the name is unusual.
    """
    known = {
        "beautifulsoup4": {"bs4"},
        "pyyaml": {"yaml"},
        "pillow": {"PIL"},
        "python-dateutil": {"dateutil"},
        "protobuf": {"google.protobuf"},
        "scikit-learn": {"sklearn"},
        "opencv-python": {"cv2"},
        "msgpack-python": {"msgpack"},
        "attrs": {"attr", "attrs"},
        "typing-extensions": {"typing_extensions"},
    }
    normalized = package.lower()
    names = {normalized, normalized.replace("-", "_")}
    names |= known.get(normalized, set())
    return {n for n in names if n}


def analyze_file(
    path: str, source: str, package: str, symbols: set[str] | None = None
) -> FileAnalysis:
    """Find imports of ``package`` and every use of what they bind, in one file."""
    analysis = FileAnalysis(path=path)
    import_names = _import_names(package)

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        # A file we cannot parse is a file we cannot clear. Recorded, not silently dropped.
        analysis.parse_failed = True
        log.debug("reachability.parse_failed", path=path, error=str(exc))
        return analysis

    lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(_module_matches(alias.name, name) for name in import_names):
                    analysis.bindings.append(
                        ImportBinding(
                            local_name=alias.asname or alias.name.split(".")[0],
                            qualified_name=alias.name,
                            line=node.lineno,
                            is_symbol=False,
                        )
                    )
                elif alias.name.split(".")[0] in _DYNAMIC_IMPORT_MARKERS:
                    analysis.has_dynamic_imports = True

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if any(_module_matches(module, name) for name in import_names):
                for alias in node.names:
                    analysis.bindings.append(
                        ImportBinding(
                            local_name=alias.asname or alias.name,
                            qualified_name=f"{module}.{alias.name}",
                            line=node.lineno,
                            is_symbol=True,
                        )
                    )
            elif module.split(".")[0] in _DYNAMIC_IMPORT_MARKERS:
                analysis.has_dynamic_imports = True

        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _DYNAMIC_IMPORT_MARKERS:
                analysis.has_dynamic_imports = True

    if not analysis.bindings:
        return analysis

    bound_names = {b.local_name: b for b in analysis.bindings}

    for node in ast.walk(tree):
        used_name: str | None = None
        qualified: str | None = None

        if isinstance(node, ast.Attribute):
            # rq.get(...) -> base name "rq", attribute "get"
            base = node.value
            if isinstance(base, ast.Name) and base.id in bound_names:
                binding = bound_names[base.id]
                used_name = node.attr
                qualified = f"{binding.qualified_name}.{node.attr}"

        elif isinstance(node, ast.Name) and node.id in bound_names:
            binding = bound_names[node.id]
            if binding.is_symbol:
                used_name = node.id
                qualified = binding.qualified_name

        if used_name is None or qualified is None:
            continue

        # When the advisory names vulnerable symbols, only those count. Otherwise any use
        # of the package is a candidate.
        if symbols and not _symbol_matches(qualified, used_name, symbols):
            continue

        line_no = getattr(node, "lineno", 0)
        snippet = lines[line_no - 1].strip() if 0 < line_no <= len(lines) else ""

        analysis.call_sites.append(
            CallSite(file_path=path, line=line_no, symbol=qualified, snippet=snippet)
        )

    # ast.walk visits Attribute and its inner Name separately, so the same source position
    # can yield duplicates. Collapse on (line, symbol).
    seen: set[tuple[int, str]] = set()
    deduped: list[CallSite] = []
    for site in analysis.call_sites:
        key = (site.line, site.symbol)
        if key not in seen:
            seen.add(key)
            deduped.append(site)
    analysis.call_sites = deduped

    return analysis


def _symbol_matches(qualified: str, used_name: str, symbols: set[str]) -> bool:
    """Match a usage against the symbol names an advisory calls out.

    Advisories are inconsistent -- some name a bare function (``get``), others a dotted
    path (``requests.sessions.Session.request``). Both the leaf and the tail of the
    qualified name are checked.
    """
    for symbol in symbols:
        leaf = symbol.rsplit(".", 1)[-1]
        if used_name == leaf or qualified.endswith(symbol):
            return True
    return False


def analyze_repository(
    files: dict[str, str],
    package: str,
    symbols: set[str] | None = None,
) -> ReachabilityResult:
    """Decide whether ``package`` is reachable across a repository's Python sources.

    ``files`` is ``{path: source}``. Non-Python paths are ignored, which is also the
    analysis's hard limit: a repository whose Python is generated, vendored, or written in
    another language cannot be cleared, and returns UNKNOWN rather than NOT_REACHABLE.
    """
    python_files = {p: c for p, c in files.items() if p.endswith(".py")}

    if not python_files:
        return ReachabilityResult(
            reachability=Reachability.UNKNOWN,
            rationale="No Python sources found to analyze, so usage could not be ruled out.",
        )

    all_sites: list[CallSite] = []
    any_bindings = False
    any_dynamic = False
    failed = 0

    for path, source in sorted(python_files.items()):
        analysis = analyze_file(path, source, package, symbols)
        if analysis.parse_failed:
            failed += 1
            continue
        any_bindings = any_bindings or bool(analysis.bindings)
        any_dynamic = any_dynamic or analysis.has_dynamic_imports
        all_sites.extend(analysis.call_sites)

    analyzed = len(python_files) - failed

    if all_sites:
        symbol_note = (
            f" matching advisory symbols {sorted(symbols)}" if symbols else ""
        )
        return ReachabilityResult(
            reachability=Reachability.REACHABLE,
            call_paths=all_sites[:20],  # enough evidence for a PR body without a wall of text
            rationale=(
                f"Found {len(all_sites)} usage site(s) of {package}{symbol_note} "
                f"across {analyzed} Python file(s)."
            ),
            files_analyzed=analyzed,
            files_failed=failed,
        )

    if any_bindings:
        # Imported but never used through a name we tracked. Common with re-exports and
        # side-effect imports, and exactly the case where a confident "no" would be wrong.
        return ReachabilityResult(
            reachability=Reachability.UNKNOWN,
            rationale=(
                f"{package} is imported but no direct usage site was found. It may be "
                f"re-exported, used dynamically, or imported for side effects."
            ),
            files_analyzed=analyzed,
            files_failed=failed,
        )

    if any_dynamic:
        return ReachabilityResult(
            reachability=Reachability.UNKNOWN,
            rationale=(
                f"No static import of {package}, but the repository uses dynamic imports, "
                f"which this analysis cannot follow."
            ),
            files_analyzed=analyzed,
            files_failed=failed,
        )

    if failed:
        return ReachabilityResult(
            reachability=Reachability.UNKNOWN,
            rationale=(
                f"No import of {package} found, but {failed} file(s) failed to parse and "
                f"could not be cleared."
            ),
            files_analyzed=analyzed,
            files_failed=failed,
        )

    return ReachabilityResult(
        reachability=Reachability.NOT_REACHABLE,
        rationale=(
            f"{package} is not imported anywhere in {analyzed} Python file(s). The "
            f"advisory does not apply to code this repository runs."
        ),
        files_analyzed=analyzed,
        files_failed=failed,
    )


def extract_advisory_symbols(advisory_details: str) -> set[str]:
    """Best-effort scrape of function names out of advisory prose.

    Advisories rarely provide machine-readable affected symbols, so this looks for
    backtick-quoted identifiers, which is the convention most write-ups follow. Purely
    additive: an empty result widens the analysis to the whole package rather than
    narrowing it to nothing.
    """
    import re

    candidates = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", advisory_details or ""))
    # Drop bare prose words that happen to be backticked.
    return {c for c in candidates if "." in c or c.islower() or "_" in c}
