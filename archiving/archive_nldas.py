"""Archive NLDAS SNOTEL-site NetCDF files into zip files grouped by site initial.

Expects files named like ``NLDAS_<site>_<start>_<end>.nc`` (e.g.
``NLDAS_elkriver_19800101_20181002.nc``). Each zip holds every site whose
name starts with the same letter, e.g. ``NLDAS_E.zip``; letters with more
than --max-group-mb of data are split into ``NLDAS_E1.zip``, ``NLDAS_E2.zip``,
... because Zenodo fails on large single uploads. A site list
(``NLDAS_site_list.txt``) giving each site's date range and archive is
uploaded after all the zips.

Each zip is written to a temporary file, uploaded to a Zenodo deposition, and
then deleted, so only one zip exists on disk at a time. The deposition is left
as an unpublished draft so it can be reviewed on the Zenodo website before
publishing (publishing cannot be undone).

Access tokens need the deposit:write scope. They are read from the
ZENODO_TOKEN (production) or ZENODO_SANDBOX_TOKEN (--sandbox) environment
variables, which can be set in the .env file at the repository root.

Usage:
    # Preview the grouping without zipping or contacting Zenodo
    python archive_nldas.py /path/to/nldas --dry-run

    # Upload to a new draft on sandbox.zenodo.org (for testing; only the
    # first 3 archives unless --limit is given)
    python archive_nldas.py /path/to/nldas --sandbox --metadata zenodo_metadata.json

    # Add to / resume an existing draft
    python archive_nldas.py /path/to/nldas --deposition-id 1234567
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv not installed: rely on the real environment
    load_dotenv = None

ZENODO_API = "https://zenodo.org/api"
ZENODO_SANDBOX_API = "https://sandbox.zenodo.org/api"


def parse_name(path):
    """Return (site, start, end) from an NLDAS_<site>_<YYYYMMDD>_<YYYYMMDD>.nc filename.

    Returns None if the filename doesn't match that pattern.
    """
    parts = path.stem.split("_")
    if len(parts) < 4 or parts[0] != "NLDAS":
        return None
    try:
        start, end = (datetime.strptime(d, "%Y%m%d").date() for d in parts[-2:])
    except ValueError:
        return None
    # Join middle parts in case a site name itself contains underscores
    return "_".join(parts[1:-2]), start, end


def group_key(site):
    first = site[0].upper()
    return first if first.isalpha() else "0-9_other"


def group_files(input_dir, pattern):
    groups = defaultdict(list)
    skipped = []
    for path in sorted(input_dir.glob(pattern)):
        parsed = parse_name(path)
        if not parsed:
            skipped.append(path)
            continue
        groups[group_key(parsed[0])].append(path)
    return groups, skipped


def split_groups(groups, max_mb):
    """Split any group over max_mb (uncompressed) into numbered parts, e.g. B -> B1, B2.

    Groups under the limit keep their plain key. Parts are balanced so each is
    roughly the same size; a single file larger than max_mb gets a part to itself.
    """
    split = {}
    for key, files in groups.items():
        sizes = [f.stat().st_size for f in files]
        total = sum(sizes)
        n = -(-total // int(max_mb * 1e6))  # ceiling division
        if n <= 1:
            split[key] = files
            continue
        # Put each file in the part its midpoint falls in along the cumulative size
        parts = [[] for _ in range(n)]
        done = 0
        for f, size in zip(files, sizes):
            parts[min(int((done + size / 2) * n / total), n - 1)].append(f)
            done += size
        parts = [p for p in parts if p]  # a very large file can leave a part empty
        width = len(str(len(parts)))  # zero-pad so B01..B12 sort correctly
        for i, part in enumerate(parts, 1):
            split[f"{key}{i:0{width}d}"] = part
    return split


def site_list_text(groups, prefix):
    """Table of site name, date range and containing archive for every file."""
    rows = []
    for key, files in groups.items():
        for f in files:
            site, start, end = parse_name(f)
            rows.append((site, start.isoformat(), end.isoformat(), f"{prefix}{key}.zip", f.name))
    rows.sort()

    header = ("site", "start_date", "end_date", "archive", "file")
    widths = [max(len(r[i]) for r in rows + [header]) for i in range(len(header))]
    lines = ["  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip()
             for row in [header] + rows]
    return "\n".join(lines) + "\n"


def write_zip(zip_path, files, compress):
    """Write files to zip_path and verify it."""
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(zip_path, "w", compression=method, allowZip64=True) as zf:
        for f in files:
            zf.write(f, arcname=f.name)

    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"CRC check failed for {bad} in {zip_path.name}")
        names = set(zf.namelist())
    missing = {f.name for f in files} - names
    if missing:
        raise RuntimeError(f"{zip_path.name} is missing {sorted(missing)}")


def md5sum(path, chunk_size=2**20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


class ZenodoError(RuntimeError):
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


class ZenodoDeposition:
    """Minimal client for a single Zenodo deposition (draft record)."""

    def __init__(self, token, sandbox=False, deposition_id=None):
        self.api = ZENODO_SANDBOX_API if sandbox else ZENODO_API
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

        if deposition_id is None:
            data = self._request("POST", f"{self.api}/deposit/depositions", json={})
        else:
            data = self._request("GET", f"{self.api}/deposit/depositions/{deposition_id}")
        self._load(data)

        if self.submitted:
            sys.exit(f"Deposition {self.id} is already published and cannot take new files. "
                     "Create a new version on Zenodo and pass its draft ID instead.")

    def _request(self, method, url, timeout=(30, 300), **kwargs):
        r = self.session.request(method, url, timeout=timeout, **kwargs)
        if not r.ok:
            raise ZenodoError(f"Zenodo {method} {url} failed ({r.status_code}): {r.text[:500]}",
                              r.status_code)
        return r.json() if r.content else None

    def _load(self, data):
        self.id = data["id"]
        self.submitted = data.get("submitted", False)
        self.bucket_url = data["links"]["bucket"]
        self.html_url = data["links"]["html"]

    @property
    def url(self):
        return f"{self.api}/deposit/depositions/{self.id}"

    def set_metadata(self, metadata):
        data = self._request("PUT", self.url, json={"metadata": metadata})
        self._load(data)

    def remote_files(self):
        """Map filename -> (file id, md5) for files already in the deposition."""
        files = self._request("GET", f"{self.url}/files")
        return {f["filename"]: (f["id"], f["checksum"]) for f in files}

    def upload(self, path, attempts=5, first_wait=60):
        """Upload path, skipping it if an identical file is already there.

        A file with the same name but different contents is replaced. Server
        errors (5xx) and dropped connections are retried, waiting first_wait
        seconds and doubling each time; before each retry the deposition is
        checked in case the file arrived despite the error.
        """
        local_md5 = md5sum(path)
        wait = first_wait
        for attempt in range(1, attempts + 1):
            try:
                existing = self.remote_files().get(path.name)
                if existing:
                    file_id, remote_md5 = existing
                    if remote_md5.removeprefix("md5:") == local_md5:
                        if attempt == 1:
                            print(f"    already on Zenodo (md5 matches), skipping upload")
                        else:
                            print(f"    arrived on Zenodo despite the error (md5 matches)")
                        return
                    print(f"    replacing existing {path.name} on Zenodo (md5 differs)")
                    self._request("DELETE", f"{self.url}/files/{file_id}")

                print(f"    uploading to Zenodo deposition {self.id} "
                      f"(attempt {attempt}/{attempts}, {datetime.now():%H:%M:%S}) ...")
                with open(path, "rb") as f:
                    # No read timeout: large uploads can take a long time
                    data = self._request("PUT", f"{self.bucket_url}/{path.name}", data=f,
                                         timeout=(30, None))
                break
            except (ZenodoError, requests.ConnectionError, requests.Timeout) as e:
                retryable = not isinstance(e, ZenodoError) or e.status >= 500
                if not retryable or attempt == attempts:
                    raise
                print(f"    {type(e).__name__}: {str(e).splitlines()[0]}")
                print(f"    retrying in {wait} s ({datetime.now():%H:%M:%S})")
                time.sleep(wait)
                wait *= 2

        remote_md5 = data["checksum"].removeprefix("md5:")
        if remote_md5 != local_md5:
            raise RuntimeError(f"Checksum mismatch after uploading {path.name}: "
                               f"local {local_md5}, Zenodo {remote_md5}")
        print(f"    uploaded (md5 {local_md5}, {datetime.now():%H:%M:%S})")


def load_metadata(path):
    with open(path) as f:
        meta = json.load(f)
    # Accept either the bare metadata dict or {"metadata": {...}}
    return meta.get("metadata", meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Folder containing the NLDAS .nc files")
    parser.add_argument("--pattern", default="NLDAS_*.nc", help="Glob for input files (default: %(default)s)")
    parser.add_argument("--prefix", default="NLDAS_", help="Prefix for zip filenames (default: %(default)s)")
    parser.add_argument("--no-compress", action="store_true",
                        help="Store files without compression (faster; useful if .nc files are already compressed)")
    parser.add_argument("--tmp-dir", type=Path,
                        help="Where to write each zip before uploading (default: system temp dir). "
                             "Needs room for the largest zip.")
    parser.add_argument("--max-group-mb", type=float, default=300,
                        help="Split letter groups larger than this many MB (uncompressed) into "
                             "numbered zips, e.g. NLDAS_B1.zip, NLDAS_B2.zip (default: %(default)s; "
                             "0 disables splitting). Zenodo rejects large single uploads.")
    parser.add_argument("--limit", type=int,
                        help="Only build the first N archives (default: 3 with --sandbox, otherwise all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the grouping without writing anything or contacting Zenodo")

    zen = parser.add_argument_group("Zenodo upload")
    zen.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org instead of zenodo.org")
    zen.add_argument("--deposition-id", type=int,
                     help="Existing draft deposition to upload to (default: create a new one)")
    zen.add_argument("--metadata", type=Path,
                     help="JSON file of Zenodo deposition metadata to apply to the draft")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Input directory not found: {args.input_dir}")

    groups, skipped = group_files(args.input_dir, args.pattern)
    if not groups:
        sys.exit(f"No files matching {args.pattern} in {args.input_dir}")
    for path in skipped:
        print(f"Skipping (unexpected filename): {path.name}")

    if args.max_group_mb:
        groups = split_groups(groups, args.max_group_mb)
    # Zenodo allows at most 100 files per record: every zip plus the site list
    if len(groups) + 1 > 100:
        sys.exit(f"{len(groups)} archives plus the site list exceeds Zenodo's 100-file limit; "
                 "raise --max-group-mb")

    limit = args.limit if args.limit is not None else (3 if args.sandbox else None)
    if limit is not None and limit < len(groups):
        # Trim before building the site list so it only covers uploaded archives
        kept = sorted(groups)[:limit]
        print(f"Limiting to {limit} of {len(groups)} archives: {', '.join(kept)}")
        groups = {k: groups[k] for k in kept}

    deposition = None
    if not args.dry_run:
        env_file = Path(__file__).resolve().parents[1] / ".env"
        if load_dotenv:
            load_dotenv(env_file)
        token_var = "ZENODO_SANDBOX_TOKEN" if args.sandbox else "ZENODO_TOKEN"
        token = os.environ.get(token_var)
        if not token:
            sys.exit(f"Set {token_var} in {env_file} or the environment to upload to Zenodo")
        # Read metadata before touching Zenodo so a bad file doesn't leave an empty draft
        metadata = load_metadata(args.metadata) if args.metadata else None
        deposition = ZenodoDeposition(token, sandbox=args.sandbox, deposition_id=args.deposition_id)
        print(f"Using Zenodo deposition {deposition.id}: {deposition.html_url}")
        if metadata:
            deposition.set_metadata(metadata)
            print(f"Applied metadata from {args.metadata}")

    for key in sorted(groups):
        files = groups[key]
        zip_name = f"{args.prefix}{key}.zip"
        size_mb = sum(f.stat().st_size for f in files) / 1e6
        print(f"{zip_name}: {len(files)} files, {size_mb:.1f} MB uncompressed")

        if args.dry_run:
            for f in files:
                print(f"    {f.name}")
            continue

        # Temp dir (rather than a bare temp file) so the zip keeps its real name for upload
        with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp:
            zip_path = Path(tmp) / zip_name
            write_zip(zip_path, files, compress=not args.no_compress)
            print(f"    zipped to {zip_path.stat().st_size / 1e6:.1f} MB")
            deposition.upload(zip_path)

    # Site list goes last so it only appears once all the zips are up
    list_name = f"{args.prefix}site_list.txt"
    list_text = site_list_text(groups, args.prefix)
    print(f"{list_name}:")
    if args.dry_run:
        print("    " + list_text.rstrip().replace("\n", "\n    "))
    else:
        with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp:
            list_path = Path(tmp) / list_name
            list_path.write_text(list_text)
            deposition.upload(list_path)

    total = sum(len(v) for v in groups.values())
    print(f"Done: {total} files in {len(groups)} archives")
    if deposition:
        expected = {f"{args.prefix}{key}.zip" for key in groups} | {list_name}
        extra = sorted(set(deposition.remote_files()) - expected)
        if extra:
            print(f"Warning: the draft also holds files not from this run: {', '.join(extra)}")
            print("    (e.g. left from an earlier grouping) -- delete them on Zenodo if unwanted.")
        print(f"Zenodo draft (not published): {deposition.html_url}")
        print("Review the record on Zenodo and publish it there when ready.")


if __name__ == "__main__":
    main()
