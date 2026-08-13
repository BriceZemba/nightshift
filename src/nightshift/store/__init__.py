"""Persistence layer.

Two backends implementing the same interface: Firestore for deployment, and a file-backed
store for offline development. Both provide the same ``try_claim_pr`` guarantee - the
right to open a pull request is granted exactly once per key - which is the property the
pipeline's resumability rests on.

The Firestore import is deliberately lazy. Importing this module must not require
``google-cloud-firestore`` to be installed, or the local path would be blocked by the very
dependency it exists to avoid.
"""

from __future__ import annotations

from typing import Any

from nightshift.config import Settings, get_settings
from nightshift.store.local import LocalStore

__all__ = ["LocalStore", "get_store"]


def get_store(settings: Settings | None = None) -> Any:
    """Return the configured store.

    ``NIGHTSHIFT_STORE=local`` selects the file-backed store, which needs no Google Cloud
    project. ``firestore`` selects the deployed backend.
    """
    settings = settings or get_settings()

    if settings.store_backend == "local":
        return LocalStore(settings.local_store_path)

    from nightshift.store.firestore import Store

    return Store()
