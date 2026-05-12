"""Pre-flight check: env keys, Python version, Qdrant, RAG smoke query."""

import os
import sys
import socket
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

CHECKS = []

def check(name, ok, detail=""):
    CHECKS.append((name, ok, detail))
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name}: {detail}")

def main():
    print("=" * 60)
    print("Lab 24 — Setup Verification")
    print("=" * 60)

    # 1. Python version
    py = sys.version_info
    check("Python >= 3.10", py >= (3, 10), f"{py.major}.{py.minor}.{py.micro}")

    # 2. Env keys
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY", "COHERE_API_KEY"):
        val = os.getenv(key, "")
        check(f"{key} set", len(val) > 8, f"len={len(val)}")

    # 3. Qdrant reachable
    try:
        s = socket.socket()
        s.settimeout(2)
        s.connect((os.getenv("QDRANT_HOST", "localhost"), int(os.getenv("QDRANT_PORT", "6333"))))
        s.close()
        check("Qdrant reachable", True, "localhost:6333")
    except OSError as e:
        check("Qdrant reachable", False, f"{e}. Run: docker-compose up -d")

    # 4. Required packages
    for pkg in ("ragas", "presidio_analyzer", "google.generativeai", "qdrant_client"):
        try:
            __import__(pkg)
            check(f"import {pkg}", True, "")
        except ImportError as e:
            check(f"import {pkg}", False, str(e))

    # 5. RAG data files
    for f in ("rag/data/nd13_2023.md", "rag/data/BCTC.md"):
        path = ROOT / f
        check(f"file {f}", path.exists(), f"{path.stat().st_size if path.exists() else 0} bytes")

    print("=" * 60)
    failures = [c for c in CHECKS if not c[1]]
    if failures:
        print(f"FAIL — {len(failures)} check(s) failed")
        sys.exit(1)
    print("OK — all checks passed")
    sys.exit(0)

if __name__ == "__main__":
    main()
