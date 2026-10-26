#!/usr/bin/env python3
"""指定した患者群の性別と年齢を集計する。患者ごとの値は出力しない。

性別と生年月日は、患者別zipに含まれるSS-MIX2の患者基本情報（ADT-00）のPID-8とPID-7から読む。
年齢は、空でない候補テキストのうち最も古い日付の時点で計算する。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

ADT_RE = re.compile(r"_ADT-00_.*\.json$")
AGE_BINS = ((None, 50, "<50"), (50, 60, "50-59"), (60, 70, "60-69"),
            (70, 80, "70-79"), (80, 90, "80-89"), (90, None, ">=90"))
SMALL_CELL = 5  # この人数未満の区分を記録する。公表の前にほかの区分と統合する


def read_sex_birth(zip_path: Path) -> tuple[str, dt.date | None, str]:
    """最後に書き込まれたADT-00から（性別、生年月日、欠損理由）を返す。"""
    if not zip_path.is_file():
        return "", None, "zip_not_found"
    with zipfile.ZipFile(zip_path) as zf:
        # ADT-00は更新のたびに追記されるため、最後の項目を使う
        entries = [i for i in zf.infolist() if ADT_RE.search(i.filename)]
        if not entries:
            return "", None, "adt_not_found"
        adt = json.loads(zf.read(entries[-1]).decode("utf-8"))
    pid = (adt.get("PID") or [{}])[0]
    sex = str(pid.get("PID-8", "")).strip()
    try:
        birth = dt.datetime.strptime(str(pid.get("PID-7", ""))[:8], "%Y%m%d").date()
    except ValueError:
        return sex, None, "invalid_birth_date"
    return sex, birth, ""


def first_candidate_date(candidate_path: Path) -> dt.date | None:
    """空でない候補テキストを持つ最も古い日付。"""
    if not candidate_path.is_file():
        return None
    data = json.loads(candidate_path.read_text(encoding="utf-8"))
    dates = [k for k, v in data.items()
             if isinstance(v, dict) and str(v.get("fam_text", "")).strip()]
    return dt.datetime.strptime(min(dates)[:8], "%Y%m%d").date() if dates else None


def age_on(birth: dt.date, day: dt.date) -> int:
    return day.year - birth.year - ((day.month, day.day) < (birth.month, birth.day))


def age_bin(age: int) -> str:
    for lower, upper, label in AGE_BINS:
        if (lower is None or age >= lower) and (upper is None or age < upper):
            return label
    raise ValueError(age)


def summarize(patient_ids: list[str], zip_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    sexes: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    ages: list[int] = []
    for patient_id in patient_ids:
        sex, birth, reason = read_sex_birth(zip_dir / f"{patient_id}.zip")
        sexes[sex or "missing"] += 1
        if reason:
            missing[reason] += 1
            continue
        day = first_candidate_date(candidate_dir / f"{patient_id}.json")
        if day is None:
            missing["candidate_date_not_found"] += 1
            continue
        ages.append(age_on(birth, day))
    bins = Counter(age_bin(a) for a in ages)
    result: dict[str, Any] = {
        "patients": len(patient_ids),
        "sex_counts": dict(sorted(sexes.items())),
        "age_reference": "earliest date with a nonempty candidate text",
        "patients_with_age": len(ages),
        "age_missing_reasons": dict(sorted(missing.items())),
        "age_bins": {label: bins.get(label, 0) for _, _, label in AGE_BINS},
    }
    if ages:
        q1, median, q3 = statistics.quantiles(ages, n=4, method="inclusive")
        result.update({
            "age_median": median, "age_q1": q1, "age_q3": q3,
            "age_mean": round(statistics.fmean(ages), 1),
            "age_sd": round(statistics.stdev(ages), 1) if len(ages) > 1 else None,
        })
    result["small_cells"] = sorted(
        [f"sex:{k}" for k, v in sexes.items() if 0 < v < SMALL_CELL]
        + [f"age:{k}" for k, v in bins.items() if 0 < v < SMALL_CELL])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", action="append", required=True, metavar="NAME=PATIENT_LIST",
                        help="集計する患者群の名前と、患者IDを1行1件で書いたファイル。複数指定できる")
    parser.add_argument("--zip-dir", type=Path, required=True,
                        help="患者別zipのディレクトリ")
    parser.add_argument("--candidate-dir", type=Path, required=True,
                        help="候補テキストのディレクトリ")
    parser.add_argument("--output", type=Path, required=True, help="集計結果を書き出すJSON")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"既存の出力を上書きしません: {args.output}")

    results = {}
    for spec in args.cohort:
        name, sep, path = spec.partition("=")
        if not sep:
            raise SystemExit(f"--cohortはNAME=PATIENT_LISTの形で指定してください: {spec}")
        ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        results[name] = summarize(ids, args.zip_dir, args.candidate_dir)

    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
