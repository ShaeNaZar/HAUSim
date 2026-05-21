import os
import sys
import tarfile
import gzip
import shutil
from pathlib import Path
import requests
from tqdm import tqdm  # pip install tqdm requests


DOI = "doi:10.7910/DVN/8SWHNO"
SERVER = "https://dataverse.harvard.edu"
OUT_DIR = Path("data/")
def get_file_list() -> list[dict]:
    """Return the list of file metadata dicts for the dataset."""
    url = f"{SERVER}/api/datasets/:persistentId/?persistentId={DOI}"
    print(f"Fetching file list from {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["data"]["latestVersion"]["files"]


def _stream_to_file(url: str, out_path: Path, size: int, desc: str) -> None:
    with requests.get(url, stream=True, timeout=300, allow_redirects=True) as r:
        if not r.ok:
            print(f"    HTTP {r.status_code}: {r.text[:400]}")
            r.raise_for_status()
        total = int(r.headers.get("content-length", size))
        with open(out_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=desc, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def download_files(dest: Path) -> None:
    """Download every file in the dataset into dest/, preserving filenames."""
    dest.mkdir(parents=True, exist_ok=True)
    files = get_file_list()
    print(f"Found {len(files)} file(s) to download.")
    for entry in files:
        df = entry["dataFile"]
        file_id = df["id"]
        filename = df["filename"]
        size = df.get("filesize", 0)
        out_path = dest / filename
        if out_path.exists() and out_path.stat().st_size == size:
            print(f"  Skip (cached): {filename}")
            continue
        print(f"  Downloading {filename} ({size / 1e6:.1f} MB)")
        # gbrecs=true auto-records any guestbook entry without blocking
        url = f"{SERVER}/api/access/datafile/{file_id}?gbrecs=true"
        try:
            _stream_to_file(url, out_path, size, filename)
        except requests.HTTPError:
            # Fall back to the un-versioned bundle download for this file
            url2 = f"{SERVER}/api/access/datafile/{file_id}?format=original&gbrecs=true"
            print(f"    Retrying with format=original …")
            _stream_to_file(url2, out_path, size, filename)
    print(f"All files saved to {dest}")


def decompress_all(root: Path) -> None:
    """
    The SLAM data ships as .tar.gz archives (one per language pair) plus
    a starter-code archive. Unpack everything in place.
    """
    # 1) Extract every .tar.gz / .tgz
    for archive in list(root.rglob("*.tar.gz")) + list(root.rglob("*.tgz")):
        print(f"Untarring {archive.name}")
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(archive.parent)

    # 2) Decompress any remaining standalone .gz files (keep originals optional)
    for gz in root.rglob("*.gz"):
        if gz.suffixes[-2:] == [".tar", ".gz"]:
            continue  # already handled above
        out = gz.with_suffix("")  # strip .gz
        if out.exists():
            continue
        print(f"Gunzipping {gz.name}")
        with gzip.open(gz, "rb") as f_in, open(out, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def main() -> None:
    download_files(OUT_DIR)
    decompress_all(OUT_DIR)

    print("\nDone. Contents:")
    for p in sorted(OUT_DIR.rglob("*"))[:40]:
        print(" ", p.relative_to(OUT_DIR))


if __name__ == "__main__":
    main()