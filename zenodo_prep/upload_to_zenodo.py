#!/usr/bin/env python3
"""
Upload the SCALES-Global bundle to Zenodo, resumably, as a DRAFT.

Run from the model root on the computer that holds the bundle:

    export ZENODO_TOKEN=...                 # personal access token
    python upload_to_zenodo.py \
        --bundle "data/outputs/zenodo" \
        --metadata zenodo_metadata.json
    # ^ defaults to the Zenodo SANDBOX. Add --production for the real archive.
    # Nothing is published: it creates/updates a DRAFT and prints its URL.
    # Review on the website, then either click Publish there or re-run with --publish.

NEW VERSION of a published record (the record keeps its concept DOI; the new
version gets its own DOI):

    python upload_to_zenodo.py --production --new-version-of NNNN \
        --bundle data/outputs/zenodo --no-metadata --replace-all

  NNNN is the record number of the published version (the number in
  10.5281/zenodo.NNNN). This creates the new-version draft (or reuses it if it
  already exists), deletes the files carried over from the previous version
  (--replace-all, only on the first run), and uploads the new bundle.
  Edit the version number and description on the website, then Publish.

Safe to re-run: it reuses the same draft (id cached beside this script) and
skips files already uploaded with a matching size. Files keep their folder
path as the Zenodo key (full/..., lite/..., uncertainty/..., tables/...);
pass --flatten if your Zenodo instance rejects '/' in file names.
"""
import argparse, json, os, sys, hashlib
from pathlib import Path
import urllib.request  # only used for a clear error if requests is missing

try:
    import requests
except Exception:
    sys.exit("ERROR: this needs the 'requests' package (pip install requests).")

SANDBOX = "https://sandbox.zenodo.org"
PRODUCTION = "https://zenodo.org"


def state_path(script_dir, base):
    tag = "sandbox" if "sandbox" in base else "production"
    return script_dir / f".zenodo_deposition_{tag}.json"


def api(method, url, token, **kw):
    kw.setdefault("params", {})["access_token"] = token
    r = requests.request(method, url, **kw)
    if r.status_code >= 400:
        sys.exit(f"Zenodo API {method} {url} -> {r.status_code}\n{r.text[:1000]}")
    return r


import time as _time


class _ProgressReader:
    """File wrapper that prints upload progress as requests reads it."""
    def __init__(self, path, size, label):
        self.f = open(path, "rb"); self.size = size; self.sent = 0
        self.label = label; self.t0 = _time.time(); self.last = 0.0
    def read(self, n=-1):
        chunk = self.f.read(n)
        self.sent += len(chunk)
        now = _time.time()
        if now - self.last > 3 or (self.size and self.sent >= self.size):
            pct = 100 * self.sent / self.size if self.size else 100.0
            rate = self.sent / 1e6 / max(now - self.t0, 1e-6)
            print(f"        {self.label}: {pct:5.1f}%  {self.sent/1e6:7.0f}/{self.size/1e6:.0f} MB  {rate:5.1f} MB/s",
                  flush=True)
            self.last = now
        return chunk
    def __len__(self):
        return self.size
    def close(self):
        try: self.f.close()
        except Exception: pass


def put_file(url, token, path, size, label, tries=6):
    """Stream one file to the Zenodo bucket with progress, a stall timeout, and retries.
    Retries transient server errors (502/503/504 and other 5xx) and network dropouts;
    4xx are treated as fatal."""
    for attempt in range(1, tries + 1):
        pr = _ProgressReader(path, size, label)
        try:
            r = requests.put(url, params={"access_token": token}, data=pr,
                             headers={"Content-Length": str(size)}, timeout=(30, 300))
            pr.close()
            if r.status_code < 400:
                return
            if r.status_code < 500 or attempt == tries:
                sys.exit(f"Zenodo PUT {url} -> {r.status_code}\n{r.text[:800]}\n"
                         f"(4xx is fatal; a 5xx that survived {tries} retries may mean Zenodo is "
                         f"struggling with the upload \u2014 try again later or from a faster network.)")
            wait = min(60, 10 * attempt)
            print(f"        Zenodo {r.status_code} on {label} (transient); "
                  f"retry {attempt+1}/{tries} in {wait}s ...", flush=True)
            _time.sleep(wait)
        except requests.exceptions.RequestException as e:
            pr.close()
            if attempt == tries:
                sys.exit(f"upload of {label} failed after {tries} tries: {e}\n"
                         f"Re-run the same command to resume (finished files are skipped).")
            wait = min(60, 10 * attempt)
            print(f"        network hiccup on {label} ({type(e).__name__}); "
                  f"retry {attempt+1}/{tries} in {wait}s ...", flush=True)
            _time.sleep(wait)


def collect_files(bundle: Path):
    """All deposit files, as (key, path). Keeps subfolder in the key."""
    out = []
    for p in sorted(bundle.rglob("*")):
        if p.is_file() and not p.name.startswith("._"):
            out.append((str(p.relative_to(bundle)), p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/outputs/zenodo")
    ap.add_argument("--metadata", default="zenodo_metadata.json")
    ap.add_argument("--production", action="store_true", help="use zenodo.org (default: sandbox)")
    ap.add_argument("--flatten", action="store_true", help="replace '/' with '__' in file keys")
    ap.add_argument("--deposition", type=int, default=None,
                    help="upload into an EXISTING deposition id (e.g. a reserved-DOI draft: the number in 10.5281/zenodo.NNNN); default creates one")
    ap.add_argument("--no-metadata", action="store_true",
                    help="do NOT touch the deposition metadata (use when you filled it in on the Zenodo website)")
    ap.add_argument("--publish", action="store_true", help="PUBLISH the draft (mints a permanent DOI)")
    ap.add_argument("--new-version-of", type=int, default=None,
                    help="record id of a PUBLISHED deposition: create (or reuse) its new-version draft")
    ap.add_argument("--replace-all", action="store_true",
                    help="delete every file already in the draft before uploading (use once, for a new version)")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("Set ZENODO_TOKEN (a Zenodo personal access token with deposit:write).")
    base = PRODUCTION if args.production else SANDBOX
    script_dir = Path(__file__).resolve().parent
    bundle = Path(args.bundle).resolve()
    if not bundle.is_dir():
        sys.exit(f"bundle not found: {bundle}")
    meta = None if args.no_metadata else json.load(open(args.metadata))

    # 1. use an explicit deposition, else reuse or create a draft
    st = state_path(script_dir, base)
    dep = None
    if args.new_version_of:
        rec = api("GET", f"{base}/api/deposit/depositions/{args.new_version_of}", token).json()
        if not rec.get("submitted"):
            sys.exit(f"record {args.new_version_of} is still an unpublished draft: upload into it with "
                     f"--deposition {args.new_version_of} instead of --new-version-of")
        latest = rec["links"].get("latest_draft", "")
        if latest.rstrip("/").endswith(f"/{args.new_version_of}"):
            r = api("POST", f"{base}/api/deposit/depositions/{args.new_version_of}/actions/newversion", token)
            latest = r.json()["links"]["latest_draft"]
            print(f"created new-version draft of record {args.new_version_of}")
        dep = api("GET", latest, token).json()
        json.dump({"id": dep["id"]}, open(st, "w"))
        print(f"using new-version draft {dep['id']}")
    elif args.deposition:
        dep = api("GET", f"{base}/api/deposit/depositions/{args.deposition}", token).json()
        print(f"using existing deposition {args.deposition}")
    elif st.exists():
        dep_id = json.load(open(st)).get("id")
        r = requests.get(f"{base}/api/deposit/depositions/{dep_id}",
                         params={"access_token": token})
        if r.status_code < 400:
            dep = r.json()
            print(f"reusing draft {dep_id}")
    if dep is None:
        dep = api("POST", f"{base}/api/deposit/depositions", token,
                  json={}, headers={"Content-Type": "application/json"}).json()
        json.dump({"id": dep["id"]}, open(st, "w"))
        print(f"created draft {dep['id']}")

    bucket = dep["links"]["bucket"]

    # 2. set metadata (skip when the user manages it on the website)
    if args.no_metadata:
        print("leaving deposition metadata as-is (--no-metadata)")
    else:
        api("PUT", dep["links"]["self"], token, json=meta,
            headers={"Content-Type": "application/json"})
        print("metadata set from " + args.metadata)

    # 3. upload files (skip ones already present with matching size)
    current = api("GET", dep["links"]["files"], token).json()
    if args.replace_all and current:
        print(f"deleting {len(current)} file(s) carried over from the previous version ...")
        for f in current:
            api("DELETE", f["links"]["self"], token)
        current = []
    existing = {f["filename"]: f.get("filesize") for f in current}
    files = collect_files(bundle)
    total = len(files)
    for i, (key, path) in enumerate(files, 1):
        upkey = key.replace("/", "__") if args.flatten else key
        size = path.stat().st_size
        if existing.get(upkey) == size:
            print(f"[{i}/{total}] skip (present) {upkey}")
            continue
        print(f"[{i}/{total}] upload {upkey} ({size/1e6:.0f} MB) ...", flush=True)
        put_file(f"{bucket}/{upkey}", token, path, size, upkey)
    print("all files uploaded")

    self_url = dep["links"].get("html") or dep["links"]["self"]
    if args.publish:
        api("POST", dep["links"]["publish"], token)
        print("PUBLISHED.")
    else:
        print("draft NOT published. Review, then publish on the site or re-run with --publish.")
    print("deposition:", self_url)


if __name__ == "__main__":
    main()
