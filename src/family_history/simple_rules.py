"""単純ルール：区切りの中の続柄とがん名の共起だけで、家族歴を抽出する。"""

from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

from .normalize import CANCER_PATTERNS, family_history_pairs, normalize_relationship, parse_strict_response
from .rule_engine import prepare_rule_input


# 最長一致で祖父の「父」、義母の「母」などを二重に拾わない。
RELATION_ALIASES = (
    "祖父", "祖母", "伯父", "叔父", "伯母", "叔母", "おじ", "おば",
    "父親", "実父", "父", "母親", "実母", "母", "実兄", "長兄", "兄",
    "実弟", "弟", "実姉", "長姉", "姉", "実妹", "妹", "実子", "子供",
    "子ども", "息子", "長男", "次男", "三男", "娘", "長女", "次女",
    "三女", "孫", "いとこ", "従兄弟", "従姉妹", "従兄妹", "従兄",
    "従弟", "従姉", "従妹", "兄弟", "姉妹", "両親",
)
_RELATIVE = "(?:" + "|".join(re.escape(s) for s in sorted(RELATION_ALIASES, key=len, reverse=True)) + ")"
SUBJECT_RE = re.compile(
    rf"(?P<nonblood>(?:夫|妻|配偶者|旦那|だんな|嫁|婿)(?:側)?の{_RELATIVE}"
    rf"|(?:義理(?:の)?|義|養|継){_RELATIVE}|義理の子|義子|養子|姑|舅|配偶者|旦那|だんな|夫|妻|嫁|婿)"
    rf"|(?P<self>患者本人|本人|患者|自分|私)"
    rf"|(?P<relative>(?:母方|父方)?{_RELATIVE}|子(?=が|は|の|に|、|:|：|\s|$))"
)
CANCER_RE = re.compile("|".join(f"(?:{pattern})" for pattern in CANCER_PATTERNS.values()))
CLAUSE_RE = re.compile(r"[^。！!、,，\n\r;；]+")
EXCLUSION_RE = re.compile(r"なし|無し|ない|ありません|否定|疑い|不明|[?？]|[（(]\s*[-−]\s*[）)]")


def _subject(match: re.Match[str] | None) -> tuple[str, str]:
    if match is None:
        return "不明", "不明"
    if match.lastgroup == "nonblood":
        return match.group(), "非血縁者"
    if match.lastgroup == "self":
        return "本人", "患者本人"
    relation = normalize_relationship(match.group())
    # 母方/父方修飾は祖父母以外でも続柄として扱う。
    if not relation:
        relation = normalize_relationship(re.sub(r"^(?:母方|父方)", "", match.group()))
    return (relation, "血縁者") if relation else ("不明", "不明")


def extract_simple_rules(text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """区切りの中に人物を表す語が1つだけあり、それが血縁者なら、同じ区切りの対象がんを採用する。

    同じ区切りに否定や疑いを表す語があれば、その区切りのがんはすべて採用しない。
    人物を表す語が2つ以上ある区切りでは採用しない。
    """
    if not isinstance(text, str):
        raise ValueError("fam_textは文字列が必要です")
    entities: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, bool]] = set()
    for clause in CLAUSE_RE.finditer(text):
        chunk = clause.group()
        subjects = list(SUBJECT_RE.finditer(chunk))
        subject = subjects[0] if len(subjects) == 1 else None
        relation, experiencer = _subject(subject)
        excluded = bool(EXCLUSION_RE.search(chunk))
        adopted = experiencer == "血縁者" and not excluded
        reason = ("clause_excluded" if excluded else
                  "blood_relative_and_cancer_in_clause" if adopted else
                  "multiple_subjects_in_clause" if len(subjects) > 1 else
                  "nonblood_self_or_unknown")
        for cancer in CANCER_RE.finditer(chunk):
            key = (relation, experiencer, cancer.group(), adopted)
            if key not in seen:
                seen.add(key)
                entities.append({"続柄": relation, "罹患者区分": experiencer,
                                 "疾患名": cancer.group(), "対象がん": True,
                                 "家族歴として採用": adopted})
            evidence.append({"relation": relation, "disease": cancer.group(),
                             "adopted": adopted, "reason": reason,
                             "cancer_span": [clause.start() + cancer.start(), clause.start() + cancer.end()],
                             "subject_span": ([clause.start() + subject.start(), clause.start() + subject.end()]
                                              if subject is not None else None)})
    prediction = {"家族歴あり": any(e["家族歴として採用"] for e in entities), "人物情報": entities}
    parse_strict_response(json.dumps(prediction, ensure_ascii=False))
    return prediction, evidence


def apply_simple_rules(records: list[dict[str, str]]) -> list[dict[str, Any]]:
    """各テキストの改行表記をそろえてから単純ルールを適用する。"""

    predictions = []
    for record in records:
        started = perf_counter()
        document, evidence = extract_simple_rules(prepare_rule_input(record["source_text"]))
        pairs = [list(pair) for pair in sorted(family_history_pairs(document))]
        predictions.append({"patient_id": record["patient_id"], "date": record["date"],
                            "source_sha256": record["source_sha256"], "status": "success", "pairs": pairs,
                            "presence": bool(pairs), "evidence": evidence,
                            "elapsed_seconds": perf_counter() - started})
    return predictions
