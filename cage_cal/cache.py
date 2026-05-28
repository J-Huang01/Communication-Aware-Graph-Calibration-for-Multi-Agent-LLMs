from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class RolloutCache:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash(key: dict[str, Any]) -> str:
        s = json.dumps(key, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]

    def _path(self, key: dict[str, Any]) -> Path:
        return self.root / f"{self._hash(key)}.json"

    def get(self, key: dict[str, Any]) -> dict[str, Any] | None:
        p = self._path(key)
        if p.exists():
            with p.open() as f:
                return json.load(f)
        return None

    def put(self, key: dict[str, Any], value: dict[str, Any]) -> None:
        p = self._path(key)
        tmp = p.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump({"key": key, "value": value}, f, ensure_ascii=False)
        tmp.replace(p)
