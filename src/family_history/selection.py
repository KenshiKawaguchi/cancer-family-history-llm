"""評価用と開発用の患者の無作為抽出。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .io_utils import read_candidates, sha256_text, write_json, write_patient_ids


def _ranked(patient_ids: list[str], seed: int) -> list[str]:
    """seedと患者IDのSHA-256の順位で並べる。乱数の実装に依存せず再現できる。"""

    prefix = f"{seed}\0".encode("utf-8")
    return sorted(patient_ids, key=lambda pid: (hashlib.sha256(prefix + pid.encode("utf-8")).digest(), pid))


def _id_text(patient_ids: list[str]) -> str:
    return "".join(f"{patient_id}\n" for patient_id in sorted(patient_ids))


def select_evaluation_patients(candidate_dir: Path, output_file: Path, *,
                               sample_size: int = 200, seed: int = 42) -> list[str]:
    """候補テキストを持つ患者から、評価用の患者を抽出する。"""

    if output_file.exists():
        raise FileExistsError(output_file)
    eligible = sorted(path.stem for path in candidate_dir.glob("*.json"))
    if len(eligible) < sample_size:
        raise ValueError(f"候補テキストを持つ患者が{sample_size}人未満です: {len(eligible)}")
    selected = sorted(_ranked(eligible, seed)[:sample_size])
    write_patient_ids(output_file, selected)
    write_json(output_file.with_suffix(".metadata.json"), {
        "seed": seed, "eligible_patients": len(eligible), "selected_patients": len(selected),
        "eligible_patient_ids_sha256": sha256_text(_id_text(eligible)),
        "patient_list_sha256": sha256_text(_id_text(selected))})
    return selected


def select_development_patients(candidate_dir: Path, evaluation_ids: set[str], output_file: Path, *,
                                sample_size: int = 200, seed: int = 43) -> list[str]:
    """評価用の患者を除き、空でない候補テキストを持つ患者から、ルール生成用の患者を抽出する。"""

    if output_file.exists():
        raise FileExistsError(output_file)
    eligible = []
    for path in sorted(candidate_dir.glob("*.json")):
        if path.stem in evaluation_ids:
            continue
        if any(row["fam_text"].strip() for row in read_candidates(path).values()):
            eligible.append(path.stem)
    if len(eligible) < sample_size:
        raise ValueError(f"評価用を除いた候補ありの患者が{sample_size}人未満です: {len(eligible)}")
    selected = sorted(_ranked(eligible, seed)[:sample_size])
    write_patient_ids(output_file, selected)
    write_json(output_file.with_suffix(".metadata.json"), {
        "seed": seed, "eligible_patients": len(eligible), "selected_patients": len(selected),
        "patient_list_sha256": sha256_text(_id_text(selected))})
    return selected
