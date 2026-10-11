# Updating a dataset

This page is for maintainers. It describes how to add a new version of one dataset to the
server and release it, without breaking the other datasets or older releases.

## How datasets are served

- Each dataset is an HDF5 file in `s3://gadopt/g-drift/` (DigitalOcean Spaces, `syd1`,
  public-read). The object name is the sha256 of the file content plus `.h5`. The
  `filename` field in `gdrift/datasets.json` gives this name.
- `gdrift/datasets.json` also lists the sha256 and the server ETag of each file.
  `load_dataset` checks the sha256 of every file. On a mismatch it downloads the file again,
  and it raises if the new download also does not match.
- A new version of a dataset has a new sha256, so it gets a new object name. Each release
  keeps its own `datasets.json`, so it keeps the object names of its own versions. For this
  reason, never delete or overwrite an object on the server.
- gdrift 0.1.3 and earlier use a different object name: `hash_name(<dataset name>) + ".h5"`.
  These objects must also stay on the server unchanged. 0.1.1 cannot load LLNL-G3D-JPS,
  MITP08, SEMUCB-WM1 and HMSL-P06, because 0.1.2 replaced these four objects
  (see [issue 41](https://github.com/g-adopt/g-drift/issues/41)).
- `tests/test_server_hashes.py` (marker `server`) compares every server ETag with the
  manifest. It also checks that the objects of 0.1.2 and 0.1.3 are unchanged, with the
  names and ETags in `tests/legacy_objects_v0.1.3.json`. The publish workflow runs it, so
  the ETags must be correct before you tag.

## Local files

- The loader caches downloads in `$GDRIFT_DATA_DIR`. If this variable is not set, it uses
  the user cache directory (`~/Library/Caches/gdrift` on macOS, `~/.cache/gdrift` on Linux).
- Before the loader downloads a file, it looks in `gdrift/data/` in the package, under the
  content name and under the name-based name of 0.1.3 and earlier. It uses a file there
  only if its sha256 matches the manifest. The conversion scripts write new files to this
  directory, so you can test a new version before you upload it.

## Things to avoid

- Do not delete or overwrite an object under `s3://gadopt/g-drift/`. Older releases can
  point to it.
- `scripts/fix_seismic_units.py` edits every seismic dataset in `gdrift/data/` in place and
  rewrites all of `datasets.json`. Its km/s test (`max < 100`) fails for a field with NaN.
- `s3cmd info` shows the md5 that s3cmd stored, not the object ETag. Read the ETag with an
  unsigned `head_object` or `curl -sI`.
- `scripts/convert_seismic_models.py` writes its output into `gdrift/data/` and
  `new_seismic_manifest_entries.json` into the working directory. Run it from the
  repository root, or give `--output-dir`.

## Procedure

1. Fix the conversion code and rebuild the file. Compare it with the old file and with the
   source.
2. Put the new sha256 into `datasets.json`, and set `filename` to `<sha256>.h5`. Set `etag`
   to `null`. Run `pytest tests -q -m "not server"` and `flake8 gdrift examples tests`.
   The tests load the new file from `gdrift/data/`.
3. Commit on a branch, push, and open a draft pull request.
4. Make sure that the content name is not on the server yet. `s3cmd put` overwrites an
   existing object without a warning. An existing object has the same content, because the
   name is its hash, but its ETag can change.

        curl -sI https://syd1.digitaloceanspaces.com/gadopt/g-drift/<sha256>.h5

   If the answer is 404, upload the new file under its content name:

        s3cmd -c ~/.s3cfg-gadopt put --acl-public --mime-type=application/x-hdf5 gdrift/data/<file> s3://gadopt/g-drift/<sha256>.h5

   `scripts/upload_singlepart.py` and `scripts/upload_to_s3.py` do the same for several
   files. They skip any object that already exists.
5. Check each new object: ETag from an unsigned `head_object`, and the sha256 of an unsigned
   download and of a download through the CDN
   (`https://gadopt.syd1.cdn.digitaloceanspaces.com/g-drift/<sha256>.h5`).
6. Put the new ETags into `datasets.json`. Run `pytest tests/test_server_hashes.py -m server`.
   Push, mark the pull request ready, wait for green CI, and merge.
7. Tag the merge commit (`vX.Y.Z`) and push the tag. The publish workflow tests, builds,
   publishes to TestPyPI and PyPI, and creates the GitHub release. Then set the notes with
   `gh release edit` and name every change since the previous tag.
8. Check the release. Install it in a new venv with `pip install gdrift==X.Y.Z`. Do not use
   an editable checkout, because it finds the files in `gdrift/data/`. Look at the
   `datasets.json` of the installed package. Then load the changed datasets with an empty
   `GDRIFT_DATA_DIR`.

## Migration to content names

`scripts/migrate_to_content_names.py` copies each name-based object to its content name on
the server. It never deletes or overwrites an object. Before it copies a file, it checks
that the ETag of the name-based object equals the `etag` in `datasets.json`.

Do the migration before you push the code change. CI starts with an empty cache, so it
downloads the content-named objects.

1. Run the script without options. It checks every dataset and prints the `s3cmd cp`
   commands. Read the list of problems. It must be empty. If a dataset has a problem, do not
   copy or overwrite anything for it. Find out why its object does not match the manifest
   first.
2. Run it with `--execute` to make the copies.
3. Run it with `--verify-download` to check the sha256 of every copy through the CDN. It must
   report 0 mismatches or errors.
4. Run it with `--update-etags`. A server-side copy can change the ETag. The script writes
   `datasets.json` only if every content-named object exists.
5. Run `pytest tests/test_server_hashes.py -m server`. Commit the ETag changes in
   `datasets.json` on the branch of the code change.
