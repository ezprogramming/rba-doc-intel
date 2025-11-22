"""MinIO-backed storage adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error

from app.config import get_settings
from app.storage.base import StorageAdapter


class MinioStorage(StorageAdapter):
    """Thin wrapper around the MinIO Python client."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self.raw_bucket = settings.minio_raw_bucket
        self.derived_bucket = settings.minio_derived_bucket

        self.ensure_bucket(self.raw_bucket)
        self.ensure_bucket(self.derived_bucket)

    def ensure_bucket(self, name: str) -> None:
        if self._client.bucket_exists(name):
            return
        self._client.make_bucket(name)

    def upload_file(self, bucket: str, object_name: str, file_path: Path) -> None:
        self._client.fput_object(bucket, object_name, str(file_path))

    def upload_fileobj(self, bucket: str, object_name: str, file_obj: BinaryIO) -> None:
        current_pos = file_obj.tell()
        file_obj.seek(0, os.SEEK_END)
        size = file_obj.tell() - current_pos
        file_obj.seek(current_pos)
        self._client.put_object(
            bucket, object_name, file_obj, length=size, part_size=5 * 1024 * 1024
        )

    def download_file(self, bucket: str, object_name: str, destination: Path) -> None:
        self._client.fget_object(bucket, object_name, str(destination))

    def object_exists(self, bucket: str, object_name: str) -> bool:
        try:
            self._client.stat_object(bucket, object_name)
            return True
        except S3Error as exc:  # pragma: no cover - network call
            if exc.code == "NoSuchKey":
                return False
            raise
