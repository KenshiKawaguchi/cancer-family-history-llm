"""正規表現ルールの検証と適用。"""

from __future__ import annotations

import copy
import re
from itertools import product
from time import perf_counter
from typing import Any

from .normalize import normalize_cancers, normalize_relationship


RULE_SCHEMA_VERSION = "qwen_pair_rules_v1"


def prepare_rule_input(text: str) -> str:
    """本文は変えず、改行の表記（<NL>、CRLF、CR）だけを改行にそろえる。"""
    return text.replace("<NL>", "\n").replace("\r\n", "\n").replace("\r", "\n")


def rule_schema() -> dict[str, Any]:
    """ルール一式のJSON Schema。"""

    fields = {
        "id": {"type": "string"}, "pattern": {"type": "string"},
        "priority": {"type": "integer"}, "reason": {"type": "string"},
        "relationships": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "cancers": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "emit_pair": {"type": "boolean"},
    }
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": [RULE_SCHEMA_VERSION]},
            "description": {"type": "string"},
            "rules": {"type": "array", "minItems": 1, "items": {
                "type": "object", "properties": copy.deepcopy(fields),
                "required": list(fields), "additionalProperties": False,
            }},
        },
        "required": ["schema_version", "description", "rules"],
        "additionalProperties": False,
    }


class PairRules:
    """ルール一式を適用し、続柄とがん種のペアを返す。

    すべての一致を集め、優先度の高い順に処理する。がん名の範囲が既に採用した一致と重なる一致は
    採用しない。emit_pairがfalseのルールが採用された範囲からは、ペアを出力しない。
    """

    def __init__(self, specification: dict[str, Any], *, text_timeout_seconds: float):
        import regex

        if specification.get("schema_version") != RULE_SCHEMA_VERSION:
            raise ValueError("schema_versionが不正です")
        self.text_timeout_seconds = text_timeout_seconds
        self.compiled = []
        for rule in specification["rules"]:
            compiled = regex.compile(rule["pattern"], regex.VERSION0)
            # 置換テンプレート（\g<relation>など）が参照するgroupを、入力の処理前に確かめる。
            names = {index: name for name, index in compiled.groupindex.items()}
            template_validator = re.compile("".join(
                f"(?P<{names[index]}>)" if index in names else "()"
                for index in range(1, compiled.groups + 1)))
            for field in ("relationships", "cancers"):
                templates = rule[field]
                if (not isinstance(templates, list) or not templates
                        or any(not isinstance(t, str) or not t.strip() for t in templates)):
                    raise ValueError(f"{rule['id']}: {field}には空でない文字列の配列が必要です")
                for template in templates:
                    try:
                        template_validator.sub(template, "")
                    except (re.error, IndexError) as exc:
                        raise ValueError(f"{rule['id']}: {field}のgroup参照が不正です: {exc}") from exc
            self.compiled.append((rule, compiled))

    def extract(self, text: str) -> tuple[set[tuple[str, str]], list[dict[str, Any]]]:
        candidates = []
        deadline = perf_counter() + self.text_timeout_seconds
        for rule, pattern in self.compiled:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"1テキストの実行上限{self.text_timeout_seconds}秒を超えました")
            for match in pattern.finditer(text, timeout=remaining):
                span = match.span("disease") if "disease" in pattern.groupindex else match.span()
                if span[0] < 0 or span[1] <= span[0]:
                    raise ValueError(f"空の一致範囲です: {rule['id']}")
                candidates.append((rule, match, span, "disease" in pattern.groupindex))
        candidates.sort(key=lambda item: -item[0]["priority"])
        accepted = []
        pairs = set()
        evidence = []
        for rule, match, (start, end), named in candidates:
            selected = not any(start < b and a < end for a, b in accepted)
            item = {"rule_id": rule["id"], "priority": rule["priority"], "reason": rule["reason"],
                    "emit_pair": rule["emit_pair"], "selected": selected,
                    "match_span": list(match.span()), "conflict_span": [start, end],
                    "conflict_span_source": "disease_group" if named else "full_match",
                    "captured_groups": list(match.groups()), "raw_pairs": [],
                    "pairs": [], "unrecognized_pairs": []}
            if selected:
                accepted.append((start, end))
                if rule["emit_pair"]:
                    relationships = [match.expand(t).strip() for t in rule["relationships"]]
                    cancers = [match.expand(t).strip() for t in rule["cancers"]]
                    for relation, cancer in product(relationships, cancers):
                        item["raw_pairs"].append([relation, cancer])
                        normalized_relation = normalize_relationship(relation)
                        normalized_cancers = normalize_cancers(cancer)
                        if not normalized_relation or not normalized_cancers:
                            item["unrecognized_pairs"].append([relation, cancer])
                        # 所定の名称に当てはまらない出力も削除せず、ペアとして残す。
                        for normalized_cancer in normalized_cancers or [cancer]:
                            pair = (normalized_relation or relation, normalized_cancer)
                            pairs.add(pair)
                            item["pairs"].append(list(pair))
            evidence.append(item)
        return pairs, evidence


def validate_rules(specification: dict[str, Any]) -> None:
    """ルール一式が書かれたとおりに実行できるかを確かめる。ルールは修正しない。"""

    schema = rule_schema()
    if set(specification) != set(schema["required"]) or not isinstance(specification["description"], str):
        raise ValueError("ルールのルート項目が不正です")
    if not isinstance(specification["rules"], list) or not specification["rules"]:
        raise ValueError("ルールが空です")
    fields = schema["properties"]["rules"]["items"]["properties"]
    for rule in specification["rules"]:
        if set(rule) != set(fields):
            raise ValueError("ルールの項目が不正です")
        for key in ("id", "pattern", "reason"):
            if not isinstance(rule[key], str) or not rule[key].strip():
                raise ValueError(f"{key}は空でない文字列が必要です")
        re.compile(rule["pattern"])
    errors = []
    for rule in specification["rules"]:
        try:
            engine = PairRules({**specification, "rules": [rule]}, text_timeout_seconds=10)
            pattern = engine.compiled[0][1]
            if type(rule["priority"]) is not int:
                raise ValueError("priorityには整数が必要です")
            if type(rule["emit_pair"]) is not bool:
                raise ValueError("emit_pairにはbooleanが必要です")
            if "disease" not in pattern.groupindex:
                raise ValueError("patternに名前付きgroup diseaseが必要です")
            if pattern.search("") is not None:
                raise ValueError("空文字に一致するルールです")
        except Exception as exc:
            errors.append(f"{rule.get('id', '(missing id)')}: {type(exc).__name__}: {exc}")
    ids = [rule["id"] for rule in specification["rules"]]
    if len(set(ids)) != len(ids):
        errors.append("ルールidが重複しています")
    if errors:
        raise ValueError("ルールを実行できません:\n" + "\n".join(errors))
    PairRules(specification, text_timeout_seconds=10)


def predict(specification: dict[str, Any], records: list[dict[str, str]],
            timeout_seconds: float) -> list[dict[str, Any]]:
    """各テキストにルールを適用する。適用中のエラーは、そのテキストの不正解として記録する。"""

    validate_rules(specification)
    engine = PairRules(specification, text_timeout_seconds=timeout_seconds)
    output = []
    for record in records:
        result = {"patient_id": record["patient_id"], "date": record["date"],
                  "source_sha256": record["source_sha256"], "status": "success"}
        started = perf_counter()
        try:
            value, evidence = engine.extract(prepare_rule_input(record["source_text"]))
            result.update(presence=bool(value), pairs=[list(pair) for pair in sorted(value)], evidence=evidence)
        except (ValueError, TimeoutError, RuntimeError) as exc:
            result.update(status="rule_error", presence=None, pairs=None, error=f"{type(exc).__name__}: {exc}")
        result["elapsed_seconds"] = perf_counter() - started
        output.append(result)
    return output
