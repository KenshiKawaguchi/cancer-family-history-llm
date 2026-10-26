"""ファイル入出力と、候補テキストの読み込み。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    """同じディレクトリの一時ファイルに書いてから置き換え、途中終了による破損を防ぐ。"""

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = handle.name
    os.replace(temporary, path)


def write_json(path: Path, data: Any) -> None:
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_sha256(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def load_patient_ids(path: Path) -> set[str]:
    """1行1件の患者ID一覧を読む。空行と#で始まる行は無視する。"""

    patient_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if value in patient_ids:
                raise ValueError(f"患者一覧に重複IDがあります: {value}")
            patient_ids.add(value)
    if not patient_ids:
        raise ValueError(f"患者一覧が空です: {path}")
    return patient_ids


def write_patient_ids(path: Path, patient_ids: set[str] | list[str]) -> None:
    write_text(path, "".join(f"{patient_id}\n" for patient_id in sorted(patient_ids)))


def read_candidates(path: Path) -> dict[str, dict[str, Any]]:
    """患者ごとの候補テキストJSON（日付 -> {"fam_text": 本文}）を読む。"""

    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"候補JSONのルートはobjectが必要です: {path}")
    for date, row in data.items():
        if (not isinstance(date, str) or not date.strip() or not isinstance(row, dict)
                or not isinstance(row.get("fam_text"), str)):
            raise ValueError(f"日付またはfam_textが不正です: {path}")
    return data


def candidate_records(candidate_dir: Path, patient_ids: set[str]) -> list[dict[str, str]]:
    """空でない候補テキストを、患者IDと日付の順に並べて返す。"""

    records = []
    for patient_id in sorted(patient_ids):
        for date, item in sorted(read_candidates(candidate_dir / f"{patient_id}.json").items()):
            text = item["fam_text"]
            if text.strip():
                records.append({"patient_id": patient_id, "date": date, "source_text": text,
                                "source_sha256": sha256_text(text.strip())})
    return records
