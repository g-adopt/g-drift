# Updating a dataset

This page is for maintainers. It describes how to replace one dataset file on the server
and release it, without breaking the other datasets.

## How datasets are served

- Each dataset is an HDF5 file in `s3://gadopt/g-drift/` (DigitalOcean Spaces, `syd1`,
  public-read). The object name is `hash_name(<dataset name>) + ".h5"`. It depends on the
  name only, not on the content.
- `gdrift/datasets.json` lists the sha256 and the server ETag of each file.
  `load_dataset` checks the sha256 of every download and raises on a mismatch.
- So when you replace a file, every older release stops with a hash error for that dataset
  (see [issue 41](https://github.com/g-adopt/g-drift/issues/41)). Say so in the release notes.
- `tests/test_server_hashes.py` (marker `server`) compares every server ETag with the
  manifest. The publish workflow runs it, so the ETags must be right before you tag.

## Things to avoid

- `scripts/upload_to_s3.py` deletes everything under `s3://gadopt/g-drift/` before it
  uploads, unless you pass `--no-delete`. Do not use it to update one dataset.
- `scripts/fix_seismic_units.py` edits every cached dataset in place and rewrites all of
  `datasets.json`. Its km/s test (`max < 100`) fails for a field with NaN.
- `s3cmd info` shows the md5 that s3cmd stored, not the object ETag. Read the ETag with an
  unsigned `head_object` or `curl -sI`.
- `scripts/convert_seismic_models.py` writes its output into `gdrift/data/` and
  `new_seismic_manifest_entries.json` into the working directory. Run it from the
  repository root, or give `--output-dir`.

## Procedure

1. Fix the conversion code and rebuild the file. Compare it with the old file and with the
   source.
2. Put the new sha256 into `datasets.json`. Run `pytest tests -q -m "not server"` and
   `flake8 gdrift examples tests`.
3. Commit on a branch, push, and open a draft pull request. Wait for green CI. The server
   is not changed yet, so the old ETags are still valid.
4. Back up each object that you will replace, and check the backup with a signed download
   against the old sha256:

        s3cmd -c ~/.s3cfg-gadopt cp --acl-public s3://gadopt/g-drift/<object> s3://gadopt/g-drift-backup-<old version>/<object>

   Write down the restore command. To get the old ETag back, upload with the old part size
   (`--multipart-chunk-size-mb`).
5. Upload only the changed objects:

        s3cmd -c ~/.s3cfg-gadopt put --acl-public --mime-type=application/x-hdf5 gdrift/data/<object> s3://gadopt/g-drift/<object>

6. Check each object: ETag from an unsigned `head_object`, and the sha256 of an unsigned
   download and of a download through the CDN
   (`https://gadopt.syd1.cdn.digitaloceanspaces.com/g-drift/<object>`).
7. Put the new ETags into `datasets.json`. Run `pytest tests/test_server_hashes.py -m server`.
   Push, mark the pull request ready, wait for green CI, and merge.
8. Tag the merge commit (`vX.Y.Z`) and push the tag. The publish workflow tests, builds,
   publishes to TestPyPI and PyPI, and creates the GitHub release. Then set the notes with
   `gh release edit` and name every change since the previous tag.
9. Check the release: `pip download gdrift==X.Y.Z --no-deps`, look at its `datasets.json`,
   delete the cached files of the changed datasets, and load them through the release.

[Pull request 35](https://github.com/g-adopt/g-drift/pull/35) followed this procedure for LLNL-G3D-JPS, MITP08, SEMUCB-WM1 and HMSL-P06 in 0.1.2.
