"""Download the four papers listed in data/sources.json into data/.

The PDFs are publicly available but are not redistributed in this repository.
Usage:  python scripts/download_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main() -> None:
    sources = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))["documents"]
    for d in sources:
        dest = DATA / d["file"]
        if dest.exists():
            print(f"exists   {dest.name}")
            continue
        print(f"download {d['url']}")
        r = requests.get(d["url"], timeout=120, headers={"User-Agent": "rag-assessment/1.0"})
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            sys.exit(f"Not a PDF: {d['url']}")
        dest.write_bytes(r.content)
        print(f"saved    {dest.name} ({len(r.content) // 1024} KB)")


if __name__ == "__main__":
    main()
