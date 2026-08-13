"""Reachability analysis.

The asymmetry under test everywhere: a wrong NOT_REACHABLE hides a real vulnerability, a
wrong UNKNOWN costs a human thirty seconds. Every ambiguous case must resolve to UNKNOWN.
"""

from __future__ import annotations

from nightshift.analysis.reachability import (
    analyze_file,
    analyze_repository,
    extract_advisory_symbols,
)
from nightshift.models import Reachability


class TestImportDetection:
    def test_plain_import(self) -> None:
        analysis = analyze_file("app.py", "import requests\nrequests.get('x')\n", "requests")
        assert len(analysis.bindings) == 1
        assert analysis.bindings[0].local_name == "requests"

    def test_aliased_import_is_resolved(self) -> None:
        source = "import requests as rq\nrq.get('x')\n"
        analysis = analyze_file("app.py", source, "requests")
        assert analysis.bindings[0].local_name == "rq"
        assert len(analysis.call_sites) == 1
        assert analysis.call_sites[0].symbol == "requests.get"

    def test_from_import_binds_the_symbol(self) -> None:
        source = "from requests import get\nget('x')\n"
        analysis = analyze_file("app.py", source, "requests")
        assert analysis.bindings[0].is_symbol is True
        assert len(analysis.call_sites) == 1

    def test_submodule_import(self) -> None:
        source = "from requests.sessions import Session\nSession()\n"
        analysis = analyze_file("app.py", source, "requests")
        assert len(analysis.call_sites) == 1
        assert "Session" in analysis.call_sites[0].symbol

    def test_similarly_named_package_is_not_matched(self) -> None:
        """requests_oauthlib starts with 'requests' but is a different distribution.
        Prefix matching without a dot boundary would produce a false positive."""
        analysis = analyze_file("app.py", "import requests_oauthlib\n", "requests")
        assert analysis.bindings == []

    def test_records_line_numbers_and_snippets(self) -> None:
        source = "import requests\n\n\nrequests.get('http://x')\n"
        analysis = analyze_file("app.py", source, "requests")
        site = analysis.call_sites[0]
        assert site.line == 4
        assert "requests.get" in site.snippet
        assert site.file_path == "app.py"

    def test_unparseable_file_is_flagged_not_dropped(self) -> None:
        analysis = analyze_file("broken.py", "def f(:\n", "requests")
        assert analysis.parse_failed is True

    def test_duplicate_positions_are_collapsed(self) -> None:
        """ast.walk visits an Attribute and its inner Name separately."""
        analysis = analyze_file("app.py", "import requests\nrequests.get('x')\n", "requests")
        assert len(analysis.call_sites) == 1


class TestDistributionNameMapping:
    def test_maps_known_distribution_to_import_name(self) -> None:
        """beautifulsoup4 imports as bs4."""
        analysis = analyze_file("app.py", "import bs4\nbs4.BeautifulSoup('')\n", "beautifulsoup4")
        assert len(analysis.call_sites) == 1

    def test_maps_pyyaml_to_yaml(self) -> None:
        analysis = analyze_file("app.py", "import yaml\nyaml.load('x')\n", "pyyaml")
        assert len(analysis.call_sites) == 1

    def test_hyphen_becomes_underscore(self) -> None:
        source = "import my_package\nmy_package.run()\n"
        analysis = analyze_file("app.py", source, "my-package")
        assert len(analysis.call_sites) == 1


class TestSymbolNarrowing:
    def test_narrows_to_named_symbol(self) -> None:
        source = "import requests\nrequests.get('x')\nrequests.post('y')\n"
        analysis = analyze_file("app.py", source, "requests", symbols={"post"})
        assert len(analysis.call_sites) == 1
        assert analysis.call_sites[0].symbol.endswith("post")

    def test_matches_dotted_symbol_path(self) -> None:
        source = "from requests.sessions import Session\nSession()\n"
        analysis = analyze_file(
            "app.py", source, "requests", symbols={"requests.sessions.Session"}
        )
        assert len(analysis.call_sites) == 1

    def test_no_symbols_means_any_usage_counts(self) -> None:
        source = "import requests\nrequests.get('x')\nrequests.post('y')\n"
        analysis = analyze_file("app.py", source, "requests", symbols=None)
        assert len(analysis.call_sites) == 2


class TestRepositoryVerdicts:
    def test_used_package_is_reachable(self) -> None:
        result = analyze_repository(
            {"app.py": "import requests\nrequests.get('x')\n"}, "requests"
        )
        assert result.reachability is Reachability.REACHABLE
        assert result.call_paths

    def test_absent_package_is_not_reachable(self) -> None:
        """The 180 -> 6 reduction. This is the only verdict allowed to be confident."""
        result = analyze_repository({"app.py": "import os\nprint(os.getcwd())\n"}, "requests")
        assert result.reachability is Reachability.NOT_REACHABLE
        assert "not imported" in result.rationale

    def test_imported_but_unused_is_unknown(self) -> None:
        """Re-exports and side-effect imports are exactly where a confident 'no' is wrong."""
        result = analyze_repository({"app.py": "import requests\n"}, "requests")
        assert result.reachability is Reachability.UNKNOWN
        assert "imported but no direct usage" in result.rationale

    def test_dynamic_imports_prevent_a_clean_no(self) -> None:
        source = "import importlib\nmod = importlib.import_module('requests')\n"
        result = analyze_repository({"app.py": source}, "requests")
        assert result.reachability is Reachability.UNKNOWN
        assert "dynamic imports" in result.rationale

    def test_unparseable_file_prevents_a_clean_no(self) -> None:
        files = {"good.py": "import os\n", "bad.py": "def f(:\n"}
        result = analyze_repository(files, "requests")
        assert result.reachability is Reachability.UNKNOWN
        assert result.files_failed == 1

    def test_no_python_sources_is_unknown(self) -> None:
        result = analyze_repository({"README.md": "# hi"}, "requests")
        assert result.reachability is Reachability.UNKNOWN

    def test_aggregates_across_files(self) -> None:
        files = {
            "a.py": "import requests\nrequests.get('x')\n",
            "b.py": "import requests\nrequests.post('y')\n",
            "c.py": "import os\n",
        }
        result = analyze_repository(files, "requests")
        assert result.reachability is Reachability.REACHABLE
        assert len(result.call_paths) == 2
        assert result.files_analyzed == 3

    def test_evidence_is_bounded(self) -> None:
        """A PR body needs enough evidence to be convincing, not a wall of text."""
        source = "import requests\n" + "\n".join(f"requests.get('{i}')" for i in range(50))
        result = analyze_repository({"app.py": source}, "requests")
        assert len(result.call_paths) == 20

    def test_non_python_files_are_ignored(self) -> None:
        files = {"app.py": "import os\n", "notes.txt": "import requests"}
        result = analyze_repository(files, "requests")
        assert result.reachability is Reachability.NOT_REACHABLE


class TestExtractAdvisorySymbols:
    def test_pulls_backticked_identifiers(self) -> None:
        details = "The `get_session` function and `requests.sessions.Session` are affected."
        symbols = extract_advisory_symbols(details)
        assert "get_session" in symbols
        assert "requests.sessions.Session" in symbols

    def test_empty_details_yields_nothing(self) -> None:
        assert extract_advisory_symbols("") == set()

    def test_no_backticks_yields_nothing(self) -> None:
        assert extract_advisory_symbols("A vulnerability exists in the parser.") == set()
