"""Tests for downloading, caching and checking dataset files (gdrift.io).

A local HTTP server stands in for the CDN and the server, so the tests need no
internet access. The file is served under its content name (`<sha256>.h5`),
as on the real server.
"""

import http.server
import os
import socket
import subprocess
import sys
import threading
import time

import h5py
import numpy
import pytest

import gdrift.io as gio
from gdrift.datasetnames import Dataset, DatasetRegistry, DatasetType


@pytest.fixture
def served_file(tmp_path):
    """Serve one small HDF5 file from a local HTTP server.

    Yields a dict with the file name, its hash ("sha256:<hex>"), its bytes,
    the base URL of the server, a base URL where nothing listens, and a list
    that records the path of every GET request. Setting `delay` in the dict
    makes the server wait that many seconds before it answers a GET.
    """
    # A small HDF5 file, named after its content as on the real server
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as f:
        f.create_dataset("data", data=numpy.arange(10.0))
    known_hash = gio.file_hash(source)
    filename = known_hash.split(":", 1)[1] + ".h5"
    served = tmp_path / "served"
    served.mkdir()
    (served / filename).write_bytes(source.read_bytes())

    info = {"gets": [], "delay": 0.0}

    class Handler(http.server.SimpleHTTPRequestHandler):
        """Quiet file server that records GET requests and can answer slowly."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(served), **kwargs)

        def do_GET(self):
            info["gets"].append(self.path)
            time.sleep(info["delay"])
            super().do_GET()

        def log_message(self, *args):
            pass

    # HTTP server on a free port, in a background thread
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # A port where nothing listens, to act as a CDN that is down
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]

    info.update({
        "filename": filename,
        "hash": known_hash,
        "content": source.read_bytes(),
        "url": f"http://127.0.0.1:{server.server_address[1]}/",
        "dead_url": f"http://127.0.0.1:{dead_port}/",
    })
    yield info
    server.shutdown()
    server.server_close()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point the cache and the package directory at empty temporary directories.

    Returns the cache directory and the package directory. Retries are
    switched off so that failing URLs fail at once.
    """
    cache = tmp_path / "cache"
    legacy = tmp_path / "package-data"
    legacy.mkdir()
    monkeypatch.setenv("GDRIFT_DATA_DIR", str(cache))
    monkeypatch.setattr(gio, "DATA_PATH", legacy)
    monkeypatch.setattr(gio, "RETRY_IF_FAILED", 0)
    return cache, legacy


def test_fetch_download_cache_repair_and_old_cache(served_file, isolated, monkeypatch):
    """The loader falls back to the second URL, reuses and repairs its cache, and finds old caches.

    Steps:
    1. The first base URL is down, so the file comes from the second one, into
       the cache directory given by GDRIFT_DATA_DIR.
    2. With both URLs down, the cached file is returned without any download.
    3. A corrupted cached file is downloaded again and replaced.
    4. With a corrupted cached file and no server, a file with the right hash
       in the package directory (under the name-based name of gdrift 0.1.3
       and earlier) is used in place.
    5. A file with the wrong hash in the package directory is ignored, and
       without a server the loader raises an error that names the dataset.
    """
    cache, legacy = isolated
    name, known_hash = served_file["filename"], served_file["hash"]

    # 1. CDN down, server up: the file is downloaded from the second URL
    monkeypatch.setattr(gio, "_base_urls", lambda: [served_file["dead_url"], served_file["url"]])
    path = gio._fetch(name, known_hash, "test-dataset")
    assert path == cache / name
    assert path.read_bytes() == served_file["content"]
    assert len(served_file["gets"]) == 1

    # 2. Both down: the cached file is returned without network access
    monkeypatch.setattr(gio, "_base_urls", lambda: [served_file["dead_url"]])
    assert gio._fetch(name, known_hash, "test-dataset") == cache / name

    # 3. A corrupted cached file is detected and downloaded again
    path.write_bytes(b"corrupted")
    monkeypatch.setattr(gio, "_base_urls", lambda: [served_file["url"]])
    assert gio._fetch(name, known_hash, "test-dataset").read_bytes() == served_file["content"]
    assert len(served_file["gets"]) == 2

    # 4. Corrupted cache, no server: a matching file in the package directory is used
    path.write_bytes(b"corrupted")
    old_name = "0" * 64 + ".h5"
    (legacy / old_name).write_bytes(served_file["content"])
    monkeypatch.setattr(gio, "_base_urls", lambda: [served_file["dead_url"]])
    assert gio._fetch(name, known_hash, "test-dataset", legacy_names=(old_name,)) == legacy / old_name

    # 5. A package-directory file with the wrong hash is not used, and the
    #    error names the dataset and the expected hash
    path.unlink()
    (legacy / old_name).write_bytes(b"old content")
    with pytest.raises(RuntimeError, match="test-dataset.*" + known_hash):
        gio._fetch(name, known_hash, "test-dataset", legacy_names=(old_name,))
    assert not (cache / name).exists()


def test_registry_download_all_permissions_and_read_only_cache(served_file, isolated, monkeypatch, tmp_path):
    """The public functions work through the registry, and file access is right.

    - `fetch_dataset` downloads a registered dataset by name.
    - The downloaded file has the permissions 0666 minus the umask (pooch
      itself writes 0600), so other users of a shared cache can read it.
    - `download_all_datasets` with a list of names counts the dataset as
      cached and returns no failures.
    - With a cache directory that cannot be created, a valid file in the
      package directory is still found, and nothing is downloaded.
    """
    cache, legacy = isolated
    monkeypatch.setattr(gio, "_base_urls", lambda: [served_file["url"]])

    # A one-entry registry in place of the real one
    ds = Dataset(name="tiny", dataset_type=DatasetType.EARTH_MODEL, source="test file",
                 file_hash=served_file["hash"], filename=served_file["filename"])
    monkeypatch.setattr(gio, "DATASET_REGISTRY", DatasetRegistry([ds]))

    path = gio.fetch_dataset("tiny")
    assert path == cache / served_file["filename"]
    assert path.stat().st_mode & 0o777 == 0o666 & ~gio._umask()
    assert gio.load_dataset("tiny")["data"].tolist() == list(range(10))

    assert gio.download_all_datasets(["tiny"]) == []
    assert len(served_file["gets"]) == 1

    # A cache directory below a read-only directory cannot be created. The
    # file in the package directory (under the name-based name) is used.
    locked_parent = tmp_path / "read-only"
    locked_parent.mkdir()
    locked_parent.chmod(0o555)
    try:
        monkeypatch.setenv("GDRIFT_DATA_DIR", str(locked_parent / "cache"))
        (legacy / (gio.hash_name("tiny") + ".h5")).write_bytes(served_file["content"])
        assert gio.fetch_dataset("tiny").parent == legacy
        assert len(served_file["gets"]) == 1
    finally:
        locked_parent.chmod(0o755)


@pytest.mark.skipif(sys.platform == "win32", reason="the download lock uses fcntl")
def test_parallel_processes_download_once(served_file, isolated):
    """Several processes that need the same missing file download it only once.

    Four processes start at the same time with an empty cache, as MPI ranks
    would. The server waits 1 s before it answers, so all four ask for the
    file while the first download is still running. With the lock, one
    process downloads and the other three find the file in the cache.
    """
    cache, _ = isolated
    served_file["delay"] = 1.0

    # Each process replaces the base URLs and switches off retries, then
    # fetches the file and prints its path
    child = (
        "import sys, gdrift.io as gio\n"
        "gio._base_urls = lambda: [sys.argv[1]]\n"
        "gio.RETRY_IF_FAILED = 0\n"
        "print(gio._fetch(sys.argv[2], sys.argv[3], 'parallel-test'))\n"
    )
    env = dict(os.environ, GDRIFT_DATA_DIR=str(cache))
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", child, served_file["url"], served_file["filename"], served_file["hash"]],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(4)
    ]
    results = [p.communicate(timeout=120) for p in procs]

    for p, (out, err) in zip(procs, results):
        assert p.returncode == 0, err
        assert out.strip() == str(cache / served_file["filename"])
    assert len(served_file["gets"]) == 1
    assert (cache / served_file["filename"]).read_bytes() == served_file["content"]
