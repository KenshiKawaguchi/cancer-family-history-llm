"""4つのプロンプト条件のプロンプトと、構造化出力のJSON Schema。"""

from __future__ import annotations

import hashlib
import json

from .constants import (
    BLOOD_RELATIONSHIPS,
    CANCER_TYPES,
    CONDITIONS,
    EXPERIENCER_TYPES,
)


# 出力に記録するプロンプトとSchemaの版。
PROMPT_VERSION = "2026-09-01-v2"
SCHEMA_VERSION = "2026-09-01-v2"


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "家族歴あり": {"type": "boolean"},
        "人物情報": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "続柄": {"type": "string"},
                    "罹患者区分": {
                        "type": "string",
                        "enum": list(EXPERIENCER_TYPES),
                    },
                    "疾患名": {"type": "string"},
                    "対象がん": {"type": "boolean"},
                    "家族歴として採用": {"type": "boolean"},
                },
                "required": [
                    "続柄",
                    "罹患者区分",
                    "疾患名",
                    "対象がん",
                    "家族歴として採用",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["家族歴あり", "人物情報"],
    "additionalProperties": False,
}


FEW_SHOT_EXAMPLES = (
    (
        "母親は子宮がんを罹患したことがある。",
        {
            "家族歴あり": True,
            "人物情報": [
                {
                    "続柄": "母",
                    "罹患者区分": "血縁者",
                    "疾患名": "子宮がん",
                    "対象がん": True,
                    "家族歴として採用": True,
                }
            ],
        },
        "主体は患者の実母で血縁者。子宮がんの家族歴として採用。",
    ),
    (
        "【家族歴】兄が肺がん、父が肝癌。",
        {
            "家族歴あり": True,
            "人物情報": [
                {
                    "続柄": "兄",
                    "罹患者区分": "血縁者",
                    "疾患名": "肺がん",
                    "対象がん": True,
                    "家族歴として採用": True,
                },
                {
                    "続柄": "父",
                    "罹患者区分": "血縁者",
                    "疾患名": "肝癌",
                    "対象がん": True,
                    "家族歴として採用": True,
                },
            ],
        },
        "兄と父はいずれも血縁者。肺がんと肝癌を各主体へ対応付け、原文表記のまま採用。",
    ),
    (
        "本人は乳癌で治療中。父が大腸がんで死亡、母は食道がん、胃がんの病歴あり。",
        {
            "家族歴あり": True,
            "人物情報": [
                {
                    "続柄": "本人",
                    "罹患者区分": "患者本人",
                    "疾患名": "乳癌",
                    "対象がん": True,
                    "家族歴として採用": False,
                },
                {
                    "続柄": "父",
                    "罹患者区分": "血縁者",
                    "疾患名": "大腸がん",
                    "対象がん": True,
                    "家族歴として採用": True,
                },
                {
                    "続柄": "母",
                    "罹患者区分": "血縁者",
                    "疾患名": "食道がん",
                    "対象がん": True,
                    "家族歴として採用": True,
                },
                {
                    "続柄": "母",
                    "罹患者区分": "血縁者",
                    "疾患名": "胃がん",
                    "対象がん": True,
                    "家族歴として採用": True,
                },
            ],
        },
        "本人の乳癌は人物情報に残すが不採用。父と母の対象がんだけを家族歴として採用。",
    ),
    (
        "症状悪化時は家族に電話するよう説明した。",
        {"家族歴あり": False, "人物情報": []},
        "家族への連絡に関する記載であり、疾患の罹患情報ではないため人物情報は空。",
    ),
    (
        "だんなの母は肺がんで他界した。",
        {
            "家族歴あり": False,
            "人物情報": [
                {
                    "続柄": "義母",
                    "罹患者区分": "非血縁者",
                    "疾患名": "肺がん",
                    "対象がん": True,
                    "家族歴として採用": False,
                }
            ],
        },
        "配偶者側の母なので非血縁者。人物情報には残すが家族歴には不採用。",
    ),
    (
        "父がBasedow病で手術をした。胃癌の既往はない。",
        {
            "家族歴あり": False,
            "人物情報": [
                {
                    "続柄": "父",
                    "罹患者区分": "血縁者",
                    "疾患名": "Basedow病",
                    "対象がん": False,
                    "家族歴として採用": False,
                },
                {
                    "続柄": "父",
                    "罹患者区分": "血縁者",
                    "疾患名": "胃癌",
                    "対象がん": True,
                    "家族歴として採用": False,
                },
            ],
        },
        "Basedow病は対象がんではなく、胃癌は否定記載なので、どちらも家族歴には不採用。",
    ),
    (
        "父か義父のどちらかが胃がんだったというが、どちらかは不明。",
        {
            "家族歴あり": False,
            "人物情報": [
                {
                    "続柄": "不明",
                    "罹患者区分": "不明",
                    "疾患名": "胃がん",
                    "対象がん": True,
                    "家族歴として採用": False,
                }
            ],
        },
        "罹患者が血縁者の父か非血縁者の義父かを特定できず、採用可否が変わる。特定人物へ推測で割り当てず、不明として不採用。",
    ),
)


def _base_system_prompt() -> str:
    relationships = "、".join(BLOOD_RELATIONSHIPS)
    cancers = "、".join(CANCER_TYPES)
    return (
        "あなたは日本語の電子カルテから、がん家族歴評価用の人物・疾患情報を抽出する専門アノテータです。\n"
        "疾患が記載された人物を、患者本人、血縁者、非血縁者、不明に分類してください。\n"
        f"血縁者の標準的な続柄は次のとおりです: {relationships}\n"
        "本人は『本人』、主体不明は『不明』。夫・妻・配偶者側または義理の親族は父母兄弟姉妹に言い換えず、原文に沿う続柄で非血縁者としてください。\n"
        f"対象がんは次の21分類に該当するがんです: {cancers}\n"
        "疾患名は原文表記を保持し、直腸癌は大腸がん、子宮頸癌は子宮がんとして対象判定してください。\n"
        "家族歴として採用をtrueにできるのは、罹患者区分が血縁者、対象がんがtrue、かつ罹患が肯定されている場合だけです。\n"
        "本人、非血縁者、不明、対象外疾患、否定、疑いも人物情報に残しますが、採用はfalseです。\n"
        "人物と疾患の対応が採否に影響するほど曖昧なら推測せず、続柄・罹患者区分を『不明』、採用falseとして残してください。\n"
        "家族歴ありは、家族歴として採用がtrueの人物情報が1件以上あればtrue、それ以外はfalseにしてください。\n"
        "同じ人物に複数疾患がある場合は疾患ごとに人物情報を分け、人物と疾患の対応を維持してください。\n"
        "JSONのみを返してください。\n"
        "疾患の罹患情報がなければ {\"家族歴あり\": false, \"人物情報\": []} を返してください。"
    )


def validate_condition(condition: str) -> str:
    if condition not in CONDITIONS:
        allowed = ", ".join(CONDITIONS)
        raise ValueError(f"未対応の実験条件です: {condition} (選択肢: {allowed})")
    return condition


def build_extraction_messages(
    text: str,
    condition: str,
    *,
    reasoning_note: str | None = None,
) -> list[dict[str, str]]:
    """構造化出力段階のメッセージを作る。FSとFS+Cではfew-shot例を、CoT条件では分析メモを加える。"""

    condition = validate_condition(condition)
    messages: list[dict[str, str]] = [{"role": "system", "content": _base_system_prompt()}]
    if condition in {"few_shot", "few_shot_cot"}:
        for example_text, example_output, _ in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example_text})
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(example_output, ensure_ascii=False),
                }
            )
    if reasoning_note:
        user_content = (
            "次の分析メモは補助情報です。必ず原文を最終的な根拠とし、誤りがあれば分析メモを採用しないでください。\n"
            f"<分析メモ>\n{reasoning_note}\n</分析メモ>\n"
            f"<原文>\n{text}\n</原文>"
        )
    else:
        user_content = text
    messages.append({"role": "user", "content": user_content})
    return messages


def build_reasoning_messages(text: str, condition: str) -> list[dict[str, str]]:
    """CoT条件の分析段階のメッセージを作る。"""

    condition = validate_condition(condition)
    if condition not in {"zero_shot_cot", "few_shot_cot"}:
        raise ValueError(f"CoT条件ではありません: {condition}")
    system_prompt = (
        "日本語の電子カルテ記載を、がん家族歴抽出のために段階的に分析してください。\n"
        "既存の家族歴抽出基準に従い、(1) 疾患記載、(2) 各疾患の主体、"
        "(3) 患者本人・血縁者・非血縁者・不明の区分、(4) 続柄、"
        "(5) 対象21がんへの該当、(6) 否定・疑い、(7) 人物と疾患の対応、"
        "(8) 家族歴として採用するか、の順に全人物を確認してください。\n"
        "複数人物のどの人の疾患か特定できず、その違いで家族歴の採用可否が変わる場合は、推測せず判定不能と明記してください。\n"
        "患者本人、非血縁者、不明、対象外疾患も分析対象から削除しないでください。\n"
        "この段階では最終JSONを出さず、簡潔な分析メモだけを返してください。"
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if condition == "few_shot_cot":
        for example_text, _, rationale in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example_text})
            messages.append({"role": "assistant", "content": rationale})
    messages.append({"role": "user", "content": text})
    return messages


def schema_sha256() -> str:
    encoded = json.dumps(
        OUTPUT_SCHEMA,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompt_sha256() -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "base_system_prompt": _base_system_prompt(),
        "few_shot_examples": FEW_SHOT_EXAMPLES,
        "zero_shot_cot": build_reasoning_messages("", "zero_shot_cot"),
        "few_shot_cot": build_reasoning_messages("", "few_shot_cot"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
