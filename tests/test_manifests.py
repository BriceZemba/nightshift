"""Manifest parsing.

The rule under test throughout: only pinned versions become dependencies, because OSV
answers questions about exact versions and a guessed version produces advisories for code
the repository does not run.
"""

from __future__ import annotations

from nightshift.models import Ecosystem
from nightshift.sources.manifests import (
    parse_package_lock,
    parse_poetry_lock,
    parse_pyproject,
    parse_requirements_txt,
    scan_files,
)


class TestRequirementsTxt:
    def test_parses_pinned_versions(self) -> None:
        scan = parse_requirements_txt("requests==2.19.1\nflask==3.0.0\n")
        assert len(scan.dependencies) == 2
        assert {d.name for d in scan.dependencies} == {"requests", "flask"}
        assert all(d.is_direct for d in scan.dependencies)

    def test_ranges_are_unresolved_not_guessed(self) -> None:
        """A range names permitted versions, not the installed one."""
        scan = parse_requirements_txt("requests>=2.0\n")
        assert scan.dependencies == []
        assert len(scan.unresolved) == 1
        assert scan.unresolved[0].name == "requests"
        assert ">=2.0" in scan.unresolved[0].spec

    def test_strips_comments_and_blank_lines(self) -> None:
        scan = parse_requirements_txt("# a comment\n\nrequests==2.19.1  # inline\n")
        assert len(scan.dependencies) == 1
        assert scan.dependencies[0].version == "2.19.1"

    def test_handles_extras(self) -> None:
        scan = parse_requirements_txt("celery[redis]==5.3.0\n")
        assert len(scan.dependencies) == 1
        assert scan.dependencies[0].name == "celery"
        assert scan.dependencies[0].version == "5.3.0"

    def test_environment_markers_do_not_skip_the_dependency(self) -> None:
        """A marker-guarded dependency may still be installed, so check it."""
        scan = parse_requirements_txt('tomli==2.0.1 ; python_version < "3.11"\n')
        assert len(scan.dependencies) == 1
        assert scan.dependencies[0].version == "2.0.1"

    def test_ignores_includes_options_and_urls(self) -> None:
        content = (
            "-r base.txt\n"
            "--index-url https://example.com/simple\n"
            "git+https://github.com/x/y.git#egg=y\n"
            "https://example.com/pkg.whl\n"
            "requests==2.19.1\n"
        )
        scan = parse_requirements_txt(content)
        assert len(scan.dependencies) == 1
        assert scan.dependencies[0].name == "requests"

    def test_strips_hashes(self) -> None:
        scan = parse_requirements_txt("requests==2.19.1 --hash=sha256:abc123\n")
        assert len(scan.dependencies) == 1
        assert scan.dependencies[0].version == "2.19.1"

    def test_normalizes_pypi_names(self) -> None:
        """PyPI treats -, _ and . as equivalent; OSV indexes the normalized form, so
        Flask_SQLAlchemy and flask-sqlalchemy must collapse to one name."""
        scan = parse_requirements_txt("Flask_SQLAlchemy==3.1.1\nZope.Interface==6.0\n")
        assert {d.name for d in scan.dependencies} == {"flask-sqlalchemy", "zope-interface"}


class TestPyproject:
    def test_pep621_ranges_are_unresolved(self) -> None:
        content = '[project]\nname = "x"\ndependencies = ["requests>=2.0", "flask~=3.0"]\n'
        scan = parse_pyproject(content)
        assert scan.dependencies == []
        assert {u.name for u in scan.unresolved} == {"requests", "flask"}

    def test_pep621_pins_are_dependencies(self) -> None:
        content = '[project]\nname = "x"\ndependencies = ["requests==2.19.1"]\n'
        scan = parse_pyproject(content)
        assert len(scan.dependencies) == 1
        assert scan.dependencies[0].version == "2.19.1"

    def test_poetry_table_is_unresolved(self) -> None:
        content = '[tool.poetry.dependencies]\npython = "^3.11"\nrequests = "^2.31"\n'
        scan = parse_pyproject(content)
        names = {u.name for u in scan.unresolved}
        assert "requests" in names
        assert "python" not in names

    def test_malformed_toml_returns_empty_not_raises(self) -> None:
        """One broken manifest must not abandon a repository that has several."""
        scan = parse_pyproject("this is not [ valid toml")
        assert scan.dependencies == []
        assert scan.unresolved == []


class TestPoetryLock:
    def test_parses_exact_versions(self) -> None:
        content = (
            '[[package]]\nname = "requests"\nversion = "2.19.1"\n\n'
            '[[package]]\nname = "urllib3"\nversion = "1.24.1"\n'
        )
        scan = parse_poetry_lock(content)
        assert len(scan.dependencies) == 2
        assert {d.version for d in scan.dependencies} == {"2.19.1", "1.24.1"}

    def test_lockfile_entries_are_transitive_by_default(self) -> None:
        """A lockfile flattens the graph; directness comes from the manifest."""
        scan = parse_poetry_lock('[[package]]\nname = "requests"\nversion = "2.19.1"\n')
        assert scan.dependencies[0].is_direct is False

    def test_skips_entries_missing_a_version(self) -> None:
        scan = parse_poetry_lock('[[package]]\nname = "broken"\n')
        assert scan.dependencies == []


class TestPackageLock:
    def test_parses_v3_packages_map(self) -> None:
        content = """
        {
          "lockfileVersion": 3,
          "packages": {
            "": {"name": "root", "version": "1.0.0"},
            "node_modules/lodash": {"version": "4.17.20"},
            "node_modules/express": {"version": "4.18.2"}
          }
        }
        """
        scan = parse_package_lock(content)
        assert {d.name for d in scan.dependencies} == {"lodash", "express"}
        assert all(d.ecosystem is Ecosystem.NPM for d in scan.dependencies)

    def test_root_entry_is_skipped(self) -> None:
        content = '{"packages": {"": {"name": "root", "version": "1.0.0"}}}'
        assert parse_package_lock(content).dependencies == []

    def test_nested_paths_are_transitive(self) -> None:
        content = """
        {
          "packages": {
            "node_modules/express": {"version": "4.18.2"},
            "node_modules/express/node_modules/debug": {"version": "2.6.9"}
          }
        }
        """
        scan = parse_package_lock(content)
        by_name = {d.name: d for d in scan.dependencies}
        assert by_name["express"].is_direct is True
        assert by_name["debug"].is_direct is False

    def test_scoped_package_name_survives(self) -> None:
        content = '{"packages": {"node_modules/@babel/core": {"version": "7.24.0"}}}'
        scan = parse_package_lock(content)
        assert scan.dependencies[0].name == "@babel/core"

    def test_parses_legacy_v1_dependency_tree(self) -> None:
        content = """
        {
          "lockfileVersion": 1,
          "dependencies": {
            "express": {
              "version": "4.18.2",
              "dependencies": {"debug": {"version": "2.6.9"}}
            }
          }
        }
        """
        scan = parse_package_lock(content)
        by_name = {d.name: d for d in scan.dependencies}
        assert by_name["express"].is_direct is True
        assert by_name["debug"].is_direct is False

    def test_malformed_json_returns_empty(self) -> None:
        assert parse_package_lock("{not json").dependencies == []


class TestScanFiles:
    def test_dispatches_on_basename_so_nested_manifests_are_found(self) -> None:
        files = {
            "services/api/requirements.txt": "requests==2.19.1\n",
            "web/package-lock.json": '{"packages": {"node_modules/lodash": {"version": "4.17.20"}}}',
        }
        scan = scan_files(files)
        assert {d.name for d in scan.dependencies} == {"requests", "lodash"}

    def test_ignores_unrecognized_files(self) -> None:
        scan = scan_files({"README.md": "# hello", "src/app.py": "print(1)"})
        assert scan.dependencies == []

    def test_deduplicates_preferring_direct(self) -> None:
        """The same package routinely appears in both a manifest and a lockfile.
        Directness changes the patch strategy, so the direct record must win."""
        files = {
            "requirements.txt": "requests==2.19.1\n",
            "poetry.lock": '[[package]]\nname = "requests"\nversion = "2.19.1"\n',
        }
        scan = scan_files(files)
        requests_deps = [d for d in scan.dependencies if d.name == "requests"]
        assert len(requests_deps) == 1
        assert requests_deps[0].is_direct is True

    def test_different_versions_are_kept_separately(self) -> None:
        files = {
            "a/requirements.txt": "requests==2.19.1\n",
            "b/requirements.txt": "requests==2.31.0\n",
        }
        scan = scan_files(files)
        assert len({d.version for d in scan.dependencies}) == 2
