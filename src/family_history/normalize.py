"""出力の検証と、続柄・がん種の正規化。"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from .constants import BLOOD_RELATIONSHIPS, CANCER_TYPES, EXPERIENCER_TYPES


NON_BLOOD_MARKERS = (
    "義",
    "夫",
    "妻",
    "旦那",
    "だんな",
    "嫁",
    "婿",
    "配偶者",
    "姑",
    "舅",
)


RELATIONSHIP_PATTERNS = {
    "祖父": r"母方祖父|父方祖父|祖父",
    "祖母": r"母方祖母|父方祖母|祖母",
    "伯父": r"伯父",
    "叔父": r"叔父|おじ",
    "伯母": r"伯母",
    "叔母": r"叔母|おば",
    "父": r"父親|実父|(?<!義)父(?!方)",
    "母": r"母親|実母|(?<!義)母(?!方)",
    "兄": r"実兄|長兄|(?<!義)兄",
    "弟": r"実弟|(?<!義)弟",
    "姉": r"実姉|長姉|(?<!義)姉",
    "妹": r"実妹|(?<!義)妹",
    "子": r"実子|子供|子ども|子(?=が|は|の|に|、|：|:)",
    "息子": r"長男|次男|三男|息子",
    "娘": r"長女|次女|三女|娘",
    "孫": r"孫",
    "いとこ": r"いとこ|従兄弟|従姉妹|従兄妹|従兄|従弟|従姉|従妹",
}


CANCER_PATTERNS = {
    "口腔・咽頭がん": r"口腔(?:がん|癌)|咽頭(?:がん|癌)|口腔・咽頭(?:がん|癌)",
    "食道がん": r"食道(?:がん|癌)",
    "胃がん": r"胃(?:がん|癌)",
    "大腸がん": r"(?:大腸|結腸|直腸|盲腸|S状結腸|横行結腸|上行結腸|下行結腸)(?:がん|癌)",
    "肝臓がん": r"(?:肝臓|肝)(?:がん|癌)|肝細胞癌",
    "胆道がん": r"(?:胆道|胆管|胆嚢|胆のう)(?:がん|癌)",
    "膵臓がん": r"(?:膵臓|膵)(?:がん|癌)",
    "喉頭がん": r"喉頭(?:がん|癌)",
    "肺がん": r"肺(?:がん|癌)",
    "皮膚がん": r"皮膚(?:がん|癌)|メラノーマ|悪性黒色腫",
    "乳房がん": r"(?:乳房|乳)(?:がん|癌)",
    "子宮がん": r"子宮(?:頸部|頸|体部|体)?(?:がん|癌)",
    "卵巣がん": r"卵巣(?:がん|癌)",
    "前立腺がん": r"前立腺(?:がん|癌)",
    "膀胱がん": r"膀胱(?:がん|癌)",
    "腎・尿路がん": r"(?:腎臓|腎|腎盂|尿管|上部尿路|腎・尿路)(?:がん|癌)",
    "脳腫瘍": r"脳腫瘍|脳(?:がん|癌)",
    "甲状腺がん": r"甲状腺(?:がん|癌)|甲状腺悪性",
    "悪性リンパ腫": r"悪性リンパ腫|リンパ腫",
    "多発性骨髄腫": r"多発性骨髄腫|骨髄腫",
    "白血病": r"白血病",
}


CANCER_ALIASES = {
    "肝がん": "肝臓がん",
    "肝癌": "肝臓がん",
    "膵がん": "膵臓がん",
    "膵癌": "膵臓がん",
    "乳がん": "乳房がん",
    "乳癌": "乳房がん",
    "子宮頸がん": "子宮がん",
    "子宮頸癌": "子宮がん",
    "子宮頸部がん": "子宮がん",
    "子宮頸部癌": "子宮がん",
    "子宮体がん": "子宮がん",
    "子宮体癌": "子宮がん",
    "子宮体部がん": "子宮がん",
    "子宮体部癌": "子宮がん",
    "腎臓がん": "腎・尿路がん",
    "腎臓癌": "腎・尿路がん",
    "腎がん": "腎・尿路がん",
    "腎癌": "腎・尿路がん",
    "腎盂がん": "腎・尿路がん",
    "尿管がん": "腎・尿路がん",
    "上部尿路がん": "腎・尿路がん",
    "腎・尿路(膀胱除く)がん": "腎・尿路がん",
    "腎・尿路（膀胱除く）がん": "腎・尿路がん",
}


STRUCTURED_ROOT_KEYS = {"家族歴あり", "人物情報"}
STRUCTURED_ENTITY_KEYS = {
    "続柄",
    "罹患者区分",
    "疾患名",
    "対象がん",
    "家族歴として採用",
}


class StructuredOutputError(ValueError):
    """出力がJSONとして解析できないか、Schemaに適合しない。"""


class SemanticOutputError(ValueError):
    """Schemaには適合するが、内容に不整合がある。"""


def parse_strict_response(raw_response: str) -> dict[str, Any]:
    """出力の文字列を修復せずにJSONとして解析し、Schemaと照合する。"""

    if not isinstance(raw_response, str) or not raw_response.strip():
        raise StructuredOutputError("JSONとして解析できません")
    try:
        decoded = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError("JSONとして解析できません") from exc
    if not isinstance(decoded, dict):
        raise StructuredOutputError("JSON Schemaに適合しません: ルートはobjectが必要です")
    if set(decoded) != STRUCTURED_ROOT_KEYS:
        raise StructuredOutputError(
            "JSON Schemaに適合しません: ルートには家族歴ありと人物情報だけが必要です"
        )
    if type(decoded["家族歴あり"]) is not bool:
        raise StructuredOutputError("JSON Schemaに適合しません: 家族歴ありはbooleanが必要です")
    entities = decoded["人物情報"]
    if not isinstance(entities, list):
        raise StructuredOutputError("JSON Schemaに適合しません: 人物情報はarrayが必要です")
    for index, entity in enumerate(entities):
        prefix = f"JSON Schemaに適合しません: 人物情報[{index}]"
        if not isinstance(entity, dict) or set(entity) != STRUCTURED_ENTITY_KEYS:
            raise StructuredOutputError(f"{prefix}の項目が不足または余分です")
        if not isinstance(entity["続柄"], str) or not entity["続柄"].strip():
            raise StructuredOutputError(f"{prefix}.続柄は空でないstringが必要です")
        if entity["罹患者区分"] not in EXPERIENCER_TYPES:
            raise StructuredOutputError(
                f"{prefix}.罹患者区分は患者本人・血縁者・非血縁者・不明のいずれかが必要です"
            )
        if not isinstance(entity["疾患名"], str) or not entity["疾患名"].strip():
            raise StructuredOutputError(f"{prefix}.疾患名は空でないstringが必要です")
        if type(entity["対象がん"]) is not bool:
            raise StructuredOutputError(f"{prefix}.対象がんはbooleanが必要です")
        if type(entity["家族歴として採用"]) is not bool:
            raise StructuredOutputError(f"{prefix}.家族歴として採用はbooleanが必要です")
    return decoded


def validate_semantic_response(response: dict[str, Any], source_text: str) -> None:
    """内容の整合性を検証する。不整合があっても出力は修正しない。

    非血縁者の続柄の扱い、採用した項目が血縁者の対象がんであること、
    続柄とがん種を所定の名称へ正規化できること、その組み合わせが原文の同じ文に近接して
    現れること、「家族歴あり」の値が採用した項目の有無と一致することを確かめる。
    """

    entities = response["人物情報"]
    adopted_count = 0
    for index, entity in enumerate(entities):
        prefix = f"人物情報[{index}]"
        relation = entity["続柄"].strip()
        experiencer = entity["罹患者区分"]
        disease = entity["疾患名"].strip()
        is_target = entity["対象がん"]
        include = entity["家族歴として採用"]

        if any(marker in relation for marker in NON_BLOOD_MARKERS):
            if experiencer != "非血縁者":
                raise SemanticOutputError(
                    f"{prefix}: 義理・配偶者側の続柄は非血縁者である必要があります"
                )
            if include:
                raise SemanticOutputError(
                    f"{prefix}: 非血縁者を家族歴として採用できません"
                )

        if not include:
            continue
        adopted_count += 1
        if experiencer != "血縁者" or not is_target:
            raise SemanticOutputError(
                f"{prefix}: 採用=trueには血縁者かつ対象がん=trueが必要です"
            )

        normalized_relation = normalize_relationship(relation)
        normalized_cancers = normalize_cancers(disease)
        if not normalized_relation or not normalized_cancers:
            raise SemanticOutputError(
                f"{prefix}: 採用対象の続柄または対象がんを正規化できません"
            )
        if not any(
            has_source_evidence(normalized_relation, cancer, source_text)
            for cancer in normalized_cancers
        ):
            raise SemanticOutputError(
                f"{prefix}: 血縁者の続柄とがんの組み合わせを原文で確認できません"
            )

    expected_family_history = adopted_count > 0
    if response["家族歴あり"] is not expected_family_history:
        raise SemanticOutputError(
            "家族歴ありは、家族歴として採用=trueの人物情報の有無と一致する必要があります"
        )


def normalize_relationship(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or any(marker in text for marker in NON_BLOOD_MARKERS):
        return ""
    replacements = {
        "母親": "母",
        "実母": "母",
        "父親": "父",
        "実父": "父",
        "実兄": "兄",
        "長兄": "兄",
        "実弟": "弟",
        "実姉": "姉",
        "長姉": "姉",
        "実妹": "妹",
        "実子": "子",
        "子供": "子",
        "子ども": "子",
        "長男": "息子",
        "次男": "息子",
        "三男": "息子",
        "長女": "娘",
        "次女": "娘",
        "三女": "娘",
        "母方祖父": "祖父",
        "父方祖父": "祖父",
        "母方祖母": "祖母",
        "父方祖母": "祖母",
        "おじ": "叔父",
        "おば": "叔母",
        "従兄弟": "いとこ",
        "従姉妹": "いとこ",
        "従兄妹": "いとこ",
        "従兄": "いとこ",
        "従弟": "いとこ",
        "従姉": "いとこ",
        "従妹": "いとこ",
    }
    text = replacements.get(text, text)
    return text if text in BLOOD_RELATIONSHIPS else ""


def _extract_cancers_from_text(text: str) -> list[str]:
    stripped = text.strip()
    if stripped in CANCER_TYPES:
        return [stripped]
    if stripped in CANCER_ALIASES:
        return [CANCER_ALIASES[stripped]]
    cancers = [name for name, pattern in CANCER_PATTERNS.items() if re.search(pattern, stripped)]
    return list(dict.fromkeys(cancers))


def normalize_cancers(value: object) -> list[str]:
    if isinstance(value, str):
        return _extract_cancers_from_text(value)
    if isinstance(value, Iterable) and not isinstance(value, (dict, bytes)):
        cancers: list[str] = []
        for item in value:
            if isinstance(item, str):
                cancers.extend(_extract_cancers_from_text(item))
        return list(dict.fromkeys(cancers))
    return []


def _segments(source_text: str) -> list[str]:
    return [
        segment.strip()
        for segment in re.split(r"[\n\r。；;]+", source_text.replace("<NL>", "\n"))
        if segment.strip()
    ]


def has_source_evidence(relationship: str, cancer: str, source_text: str) -> bool:
    relationship_pattern = RELATIONSHIP_PATTERNS.get(relationship)
    cancer_pattern = CANCER_PATTERNS.get(cancer)
    if not relationship_pattern or not cancer_pattern:
        return False
    non_blood_pattern = re.compile(r"義|夫の|妻の|旦那の|だんなの|配偶者の|嫁の|婿の|姑|舅")
    for segment in _segments(source_text):
        relationship_matches = list(re.finditer(relationship_pattern, segment))
        cancer_matches = list(re.finditer(cancer_pattern, segment))
        if not relationship_matches or not cancer_matches:
            continue
        pairs = sorted(
            (
                abs(rel.start() - disease.start()),
                rel,
                disease,
            )
            for rel in relationship_matches
            for disease in cancer_matches
        )
        for distance, rel_match, cancer_match in pairs:
            if distance > 100:
                continue
            start = max(0, min(rel_match.start(), cancer_match.start()) - 6)
            end = min(len(segment), max(rel_match.end(), cancer_match.end()) + 6)
            if non_blood_pattern.search(segment[start:end]):
                continue
            return True
    return False


def family_history_pairs(document: dict[str, Any]) -> set[tuple[str, str]]:
    """構造化出力から、家族歴として採用された血縁者の続柄とがん種のペアを取り出す。"""

    pairs: set[tuple[str, str]] = set()
    for entity in document.get("人物情報", []):
        if (entity.get("罹患者区分") != "血縁者" or entity.get("対象がん") is not True
                or entity.get("家族歴として採用") is not True):
            continue
        relationship = normalize_relationship(entity.get("続柄"))
        if not relationship:
            continue
        for cancer in normalize_cancers(entity.get("疾患名")):
            pairs.add((relationship, cancer))
    return pairs


def strict_pairs(document: dict[str, Any]) -> list[list[str]]:
    """Schemaに適合する出力からペアを取り出し、「家族歴あり」の値との一致も確かめる。"""

    parsed = parse_strict_response(json.dumps(document, ensure_ascii=False))
    pairs = [list(pair) for pair in sorted(family_history_pairs(parsed))]
    if parsed["家族歴あり"] != bool(pairs):
        raise ValueError("家族歴ありの値と採用したペアが一致しません")
    return pairs
