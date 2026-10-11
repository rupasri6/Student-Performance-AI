"""Download the public UCI Student Performance zip into data/raw if files are missing."""

from __future__ import annotations

import io
import zipfile
from urllib.request import urlopen

from src.config import DATA_RAW, MATH_CSV, POR_CSV

UCI_ZIP = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def download_if_needed() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if MATH_CSV.exists() and POR_CSV.exists():
        print("Raw CSVs already present in", DATA_RAW)
        return
    print("Downloading UCI Student Performance dataset...")
    with urlopen(UCI_ZIP, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(DATA_RAW)
        # The zip contains student.zip; extract nested csvs if needed.
        nested = DATA_RAW / "student.zip"
        if nested.exists():
            with zipfile.ZipFile(nested) as inner:
                inner.extractall(DATA_RAW)
    if not MATH_CSV.exists() or not POR_CSV.exists():
        raise RuntimeError("Download finished but student-mat.csv / student-por.csv were not found.")
    print("Saved", MATH_CSV.name, "and", POR_CSV.name)


if __name__ == "__main__":
    download_if_needed()
