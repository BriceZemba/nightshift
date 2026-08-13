"""Manifest and lockfile parsing - repository contents to a concrete dependency list.

The governing constraint is that OSV answers questions about *exact versions*. A spec like
``requests>=2.0`` names a range, not an installed artifact, and guessing which version sits
in that range would produce advisories for code the repository does not actually run. So
this module only emits a :class:`Dependency` when the version is pinned, and reports
everything else as :class:`UnresolvedSpec` for honest accounting.

That ordering is why lockfiles are preferred over manifests: ``poetry.lock`` and
``package-lock.json`` describe what is installed, while ``pyproject.toml`` describes what
is permitted.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field

import structlog

from nightshift.models import Dependency, Ecosystem

log = structlog.get_logger(__name__)

#: Filenames recognized, most authoritative first. The first lockfile found for an
#: ecosystem wins; manifests are only consulted to learn which dependencies are direct.
LOCKFILES = ("poetry.lock", "uv.lock", "package-lock.json")
MANIFESTS = ("requirements.txt", "pyproject.toml", "package.json")

#: A pinned requirement line: name, optional extras, "==", version.
#: Deliberately does not accept ">=", "~=" or "^" - those are ranges, not versions.
_PINNED_REQUIREMENT = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[(?P<extras>[^\]]*)\])?"
    r"\s*==\s*"
    r"(?P<version>[A-Za-z0-9][A-Za-z0-9.*+!-]*)"
)

#: Any requirement line, pinned or not, used to report unresolved specs.
_ANY_REQUIREMENT = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[(?P<extras>[^\]]*)\])?"
    r"\s*(?P<spec>[<>=!~^].*)?$"
)


@dataclass(frozen=True)
class UnresolvedSpec:
    """A dependency whose installed version could not be determined.

    Surfaced rather than dropped: "we could not check 40 of your dependencies because you
    have no lockfile" is useful information, and quietly ignoring them would overstate
    coverage.
    """

    name: str
    ecosystem: Ecosystem
    spec: str
    manifest_path: str
    reason: str


@dataclass
class ManifestScan:
    dependencies: list[Dependency] = field(default_factory=list)
    unresolved: list[UnresolvedSpec] = field(default_factory=list)

    def extend(self, other: ManifestScan) -> None:
        self.dependencies.extend(other.dependencies)
        self.unresolved.extend(other.unresolved)

    def deduplicate(self) -> ManifestScan:
        """Collapse duplicates, preferring direct declarations over transitive ones.

        The same package routinely appears in both a manifest and a lockfile. Directness
        changes the patch strategy - a direct dependency can be bumped in place, while a
        transitive one may need a constraint pin - so the direct record is the one to keep.
        """
        best: dict[tuple[str, Ecosystem, str], Dependency] = {}
        for dep in self.dependencies:
            key = (dep.name.lower(), dep.ecosystem, dep.version)
            existing = best.get(key)
            if existing is None or (dep.is_direct and not existing.is_direct):
                best[key] = dep
        return ManifestScan(dependencies=list(best.values()), unresolved=self.unresolved)


def _normalize_pypi_name(name: str) -> str:
    """PyPI treats runs of ``-``, ``_`` and ``.`` as equivalent and is case-insensitive.

    OSV indexes the normalized form, so ``Flask_SQLAlchemy`` and ``flask-sqlalchemy`` must
    collapse to one name or the same package gets queried twice and matched zero times.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements_txt(content: str, path: str = "requirements.txt") -> ManifestScan:
    """Parse a pip requirements file.

    Ignores ``-r``/``-c`` includes, options, URLs and editable installs. Environment
    markers are stripped: a dependency guarded by ``; python_version < "3.10"`` may still
    be installed, so it is safer to check it than to skip it.
    """
    scan = ManifestScan()

    for raw_line in content.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        # Options, includes, editable installs and direct URLs are out of scope.
        if line.startswith("-") or "://" in line or line.startswith("git+"):
            continue

        # Strip environment markers and inline hashes.
        line = line.split(";", 1)[0].strip()
        line = line.split(" --hash", 1)[0].strip()
        if not line:
            continue

        pinned = _PINNED_REQUIREMENT.match(line)
        if pinned:
            scan.dependencies.append(
                Dependency(
                    name=_normalize_pypi_name(pinned.group("name")),
                    ecosystem=Ecosystem.PYPI,
                    version=pinned.group("version"),
                    manifest_path=path,
                    is_direct=True,
                )
            )
            continue

        loose = _ANY_REQUIREMENT.match(line)
        if loose:
            scan.unresolved.append(
                UnresolvedSpec(
                    name=_normalize_pypi_name(loose.group("name")),
                    ecosystem=Ecosystem.PYPI,
                    spec=(loose.group("spec") or "").strip(),
                    manifest_path=path,
                    reason="version is a range, not a pin; add a lockfile for exact versions",
                )
            )

    return scan


def parse_pyproject(content: str, path: str = "pyproject.toml") -> ManifestScan:
    """Parse PEP 621 ``[project].dependencies`` and Poetry's dependency table.

    Almost everything here lands in ``unresolved``: pyproject declares constraints, not
    installed versions. It is parsed to learn which dependencies are *direct*, which the
    lockfile alone cannot tell us.
    """
    scan = ManifestScan()

    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        log.warning("manifest.invalid_toml", path=path, error=str(exc))
        return scan

    for raw in data.get("project", {}).get("dependencies", []):
        if not isinstance(raw, str):
            continue
        stripped = raw.split(";", 1)[0].strip()
        pinned = _PINNED_REQUIREMENT.match(stripped)
        if pinned:
            scan.dependencies.append(
                Dependency(
                    name=_normalize_pypi_name(pinned.group("name")),
                    ecosystem=Ecosystem.PYPI,
                    version=pinned.group("version"),
                    manifest_path=path,
                    is_direct=True,
                )
            )
        else:
            loose = _ANY_REQUIREMENT.match(stripped)
            if loose:
                scan.unresolved.append(
                    UnresolvedSpec(
                        name=_normalize_pypi_name(loose.group("name")),
                        ecosystem=Ecosystem.PYPI,
                        spec=(loose.group("spec") or "").strip(),
                        manifest_path=path,
                        reason="declared constraint, not an installed version",
                    )
                )

    poetry_deps = (
        data.get("tool", {}).get("poetry", {}).get("dependencies", {})
        if isinstance(data.get("tool"), dict)
        else {}
    )
    for name, spec in poetry_deps.items():
        if name.lower() == "python":
            continue
        spec_text = spec if isinstance(spec, str) else json.dumps(spec)
        scan.unresolved.append(
            UnresolvedSpec(
                name=_normalize_pypi_name(name),
                ecosystem=Ecosystem.PYPI,
                spec=spec_text,
                manifest_path=path,
                reason="poetry constraint; resolve via poetry.lock",
            )
        )

    return scan


def parse_poetry_lock(content: str, path: str = "poetry.lock") -> ManifestScan:
    """Parse ``poetry.lock``. Authoritative - every entry is an exact installed version."""
    scan = ManifestScan()

    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        log.warning("manifest.invalid_toml", path=path, error=str(exc))
        return scan

    for package in data.get("package", []):
        name = package.get("name")
        version = package.get("version")
        if not name or not version:
            continue
        scan.dependencies.append(
            Dependency(
                name=_normalize_pypi_name(name),
                ecosystem=Ecosystem.PYPI,
                version=version,
                manifest_path=path,
                # A lockfile flattens the graph; directness comes from the manifest.
                is_direct=False,
            )
        )

    return scan


def parse_uv_lock(content: str, path: str = "uv.lock") -> ManifestScan:
    """Parse ``uv.lock``. Same ``[[package]]`` shape as poetry.lock."""
    scan = parse_poetry_lock(content, path=path)
    return scan


def parse_package_lock(content: str, path: str = "package-lock.json") -> ManifestScan:
    """Parse npm's ``package-lock.json`` (lockfileVersion 2 or 3).

    The ``packages`` map keys paths like ``node_modules/foo`` and
    ``node_modules/foo/node_modules/bar``; the package name is the segment after the final
    ``node_modules/``. Scoped packages (``@scope/name``) survive that rule intact. The
    root entry has an empty key and is skipped - it is the project itself.
    """
    scan = ManifestScan()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        log.warning("manifest.invalid_json", path=path, error=str(exc))
        return scan

    packages = data.get("packages")
    if isinstance(packages, dict):
        for location, meta in packages.items():
            if not location or not isinstance(meta, dict):
                continue
            version = meta.get("version")
            if not version:
                continue
            _, _, name = location.rpartition("node_modules/")
            if not name:
                continue
            scan.dependencies.append(
                Dependency(
                    name=name,
                    ecosystem=Ecosystem.NPM,
                    version=version,
                    manifest_path=path,
                    # Top-level "node_modules/x" is direct; nested paths are transitive.
                    is_direct=location.count("node_modules/") == 1,
                )
            )
        return scan

    # lockfileVersion 1 used a nested "dependencies" tree instead.
    def walk(tree: dict[str, object], *, direct: bool) -> None:
        for name, meta in tree.items():
            if not isinstance(meta, dict):
                continue
            version = meta.get("version")
            if isinstance(version, str):
                scan.dependencies.append(
                    Dependency(
                        name=name,
                        ecosystem=Ecosystem.NPM,
                        version=version,
                        manifest_path=path,
                        is_direct=direct,
                    )
                )
            nested = meta.get("dependencies")
            if isinstance(nested, dict):
                walk(nested, direct=False)

    legacy = data.get("dependencies")
    if isinstance(legacy, dict):
        walk(legacy, direct=True)

    return scan


#: Filename to parser. Used by :func:`scan_files`.
PARSERS = {
    "requirements.txt": parse_requirements_txt,
    "pyproject.toml": parse_pyproject,
    "poetry.lock": parse_poetry_lock,
    "uv.lock": parse_uv_lock,
    "package-lock.json": parse_package_lock,
}


def scan_files(files: dict[str, str]) -> ManifestScan:
    """Parse every recognized manifest in ``{path: content}``.

    Unrecognized files are ignored. Matching is on the basename, so nested manifests such
    as ``services/api/requirements.txt`` are picked up too.
    """
    scan = ManifestScan()

    for path, content in files.items():
        basename = path.replace("\\", "/").rsplit("/", 1)[-1]
        parser = PARSERS.get(basename)
        if parser is None:
            continue
        try:
            scan.extend(parser(content, path))
        except (ValueError, TypeError, KeyError) as exc:
            # One malformed manifest must not abandon a repository that has several.
            log.warning("manifest.parse_failed", path=path, error=str(exc))

    result = scan.deduplicate()
    log.info(
        "manifest.scanned",
        files=len(files),
        dependencies=len(result.dependencies),
        unresolved=len(result.unresolved),
    )
    return result


def manifest_paths_to_fetch() -> tuple[str, ...]:
    """Filenames worth requesting from a repository."""
    return LOCKFILES + MANIFESTS
