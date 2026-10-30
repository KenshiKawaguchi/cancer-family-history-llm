"""出力の形式の確認と、続柄・がん種の正規化（辞書v2）。

出力は形式だけを確認する（JSONとして解析でき、Schemaに適合し、続柄と疾患名が空でないこと）。
人物や文脈の判断に基づいて採用項目を除いたり訂正したりはしない。
辞書v2は、評価対象と重ならないルール生成用の200患者で見つかった表記ゆれの型から一般化した。
"""

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


# 臓器名と「癌」の間に、部位・病期・組織型を表す語が入ってもよい。
# 「肛門」で始まる部位（直腸肛門部など）は大腸に含めない。
CANCER_SUFFIX = r"(?:がん|癌|ガン|悪性腫瘍|Ca(?![a-z]))"
CANCER_MODIFIER = (
    r"(?:(?!肛門)[一-龥A-Za-z]{1,4}部"
    r"|早期|進行|局所進行|再発|末期|表在|粘膜内|浸潤性|浸潤"
    r"|腺|扁平上皮|小細胞|非小細胞|大細胞|印環細胞|粘液|乳頭|小葉|乳管|未分化"
    r"|[低中高]分化型?腺?)"
)


def _organ_cancer(organs: str) -> str:
    return rf"(?:{organs})(?:{CANCER_MODIFIER}){{0,3}}{CANCER_SUFFIX}"


CANCER_PATTERNS = {
    "口腔・咽頭がん": _organ_cancer(r"口腔・咽頭|口腔|咽頭"),
    "食道がん": _organ_cancer(r"食道"),
    "胃がん": _organ_cancer(r"胃"),
    "大腸がん": _organ_cancer(
        r"大腸|S状結腸|横行結腸|上行結腸|下行結腸|結腸|直腸|盲腸|回盲部"
    ),
    "肝臓がん": _organ_cancer(r"肝臓|肝") + r"|肝細胞癌|(?<![A-Za-z])HCC(?![A-Za-z])",
    "胆道がん": _organ_cancer(r"胆道|胆管|胆嚢|胆のう"),
    "膵臓がん": _organ_cancer(r"膵臓|膵"),
    "喉頭がん": _organ_cancer(r"喉頭"),
    "肺がん": _organ_cancer(r"肺"),
    "皮膚がん": _organ_cancer(r"皮膚") + r"|メラノーマ|悪性黒色腫",
    "乳房がん": _organ_cancer(r"乳房|乳(?!頭)"),
    "子宮がん": _organ_cancer(r"子宮(?:頸部|頸|体部|体)?"),
    "卵巣がん": _organ_cancer(r"卵巣"),
    "前立腺がん": _organ_cancer(r"前立腺"),
    "膀胱がん": _organ_cancer(r"膀胱"),
    "腎・尿路がん": _organ_cancer(r"腎臓|腎盂|腎・尿路|腎|尿管|上部尿路"),
    "脳腫瘍": r"脳腫瘍|" + _organ_cancer(r"脳"),
    "甲状腺がん": _organ_cancer(r"甲状腺") + r"|甲状腺悪性",
    "悪性リンパ腫": r"悪性リンパ腫|リンパ腫|(?<![A-Za-z])DLBCL(?![A-Za-z])",
    "多発性骨髄腫": r"多発性骨髄腫|骨髄腫",
    "白血病": r"白血病|(?<![A-Za-z])(?:AML|ALL|CML|CLL)(?![A-Za-z])",
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

# 正規化できない採用項目を、誤って抽出されたペアとして数えるための印。
UNNORMALIZED = "正規化できない採用項目"


class StructuredOutputError(ValueError):
    """出力がJSONとして解析できないか、Schemaに適合しない。"""


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
    # 先頭の「母方（の）」「父方（の）」を外し、「父の兄」などを伯父・伯母・叔父・叔母に読み替える。
    text = re.sub(r"^(?:母方|父方)の?(?=.)", "", text)
    parent_sibling = re.fullmatch(r"(?:父|母)の(兄|姉|弟|妹)", text)
    if parent_sibling:
        text = {"兄": "伯父", "姉": "伯母", "弟": "叔父", "妹": "叔母"}[parent_sibling.group(1)]
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


def family_history_pairs(document: dict[str, Any]) -> set[tuple[str, str]]:
    """構造化された記録から、家族歴として採用された血縁者の続柄とがん種のペアを取り出す。

    人手正解と単純ルールの出力に用いる。
    """

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


def adopted_pairs(document: dict[str, Any]) -> set[tuple[str, str]]:
    """LLMの出力から、「家族歴として採用」がtrueの項目をすべてペアとして取り出す。

    罹患者区分や「対象がん」の値にかかわらず、採用された項目はすべて含める。
    続柄または疾患名を正規化できない採用項目も除かず、正解と一致しないペアとして残す。
    """

    pairs: set[tuple[str, str]] = set()
    for index, entity in enumerate(document.get("人物情報", [])):
        if entity.get("家族歴として採用") is not True:
            continue
        relation = str(entity.get("続柄", "")).strip()
        disease = str(entity.get("疾患名", "")).strip()
        relationship = normalize_relationship(relation)
        cancers = normalize_cancers(disease)
        if relationship and cancers:
            pairs.update((relationship, cancer) for cancer in cancers)
        else:
            pairs.add((UNNORMALIZED, f"{index}:{relation}:{disease}"))
    return pairs
