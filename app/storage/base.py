"""Abstract base classes and helpers for storage adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO


class StorageAdapter(ABC):
    """Define a minimal interface for MinIO/S3-compatible storage."""

    @abstractmethod
    def ensure_bucket(self, name: str) -> None:  # pragma: no cover - interface contract
        """Ensure a bucket exists (idempotent)."""

    @abstractmethod
    def upload_file(self, bucket: str, object_name: str, file_path: Path) -> None:
        """Upload a local file to the target bucket/key."""

    @abstractmethod
    def upload_fileobj(self, bucket: str, object_name: str, file_obj: BinaryIO) -> None:
        """Upload a file-like object without touching disk."""

    @abstractmethod
    def download_file(self, bucket: str, object_name: str, destination: Path) -> None:
        """Download an object to the provided destination path."""

    @abstractmethod
    def object_exists(self, bucket: str, object_name: str) -> bool:
        """Return True if an object exists."""
