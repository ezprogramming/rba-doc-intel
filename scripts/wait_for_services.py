"""Simple readiness probe for Postgres and MinIO."""

from __future__ import annotations

import os
import sys
import time
from contextlib import suppress

import psycopg
from minio import Minio

POSTGRES_DSN = os.environ.get("POSTGRES_DSN") or os.environ.get("DATABASE_URL")
if POSTGRES_DSN and "+psycopg" in POSTGRES_DSN:
    POSTGRES_DSN = POSTGRES_DSN.replace("+psycopg", "")
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.environ.get("MINIO_SECURE", "0") not in {"0", "false", "False"}


def wait_for_postgres(timeout: int = 60) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with psycopg.connect(POSTGRES_DSN, connect_timeout=5):
                print("Postgres is ready", flush=True)
                return
        except Exception as exc:  # noqa: BLE001
            print(f"Waiting for Postgres: {exc}", flush=True)
            time.sleep(2)
    raise TimeoutError("Postgres not ready within timeout")


def wait_for_minio(timeout: int = 60) -> None:
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )
    start = time.time()
    while time.time() - start < timeout:
        with suppress(Exception):
            client.list_buckets()
            print("MinIO is ready", flush=True)
            return
        print("Waiting for MinIO...", flush=True)
        time.sleep(2)
    raise TimeoutError("MinIO not ready within timeout")


def main() -> None:
    wait_for_postgres()
    wait_for_minio()


if __name__ == "__main__":
    try:
        main()
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
