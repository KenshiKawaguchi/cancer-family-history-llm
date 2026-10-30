"""Qwen3.5による正規表現ルールの生成。

ルール生成用の患者の候補テキストにQwen3.5のFSを適用した抽出結果を疑似正解とする。
原文と疑似正解の例から初期ルールを生成させ、誤分類例と正解例を与えて1回だけ作り直させる。
作り直したルールを最終版とする。実行できないルールは除外し、ルールは修正しない。
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from .constants import BLOOD_RELATIONSHIPS, CANCER_TYPES
from .inference import GENERATION_CONFIG, ModelSettings, extract_content, load_model, usage_of
from .io_utils import candidate_records, json_sha256, load_json, load_patient_ids, write_json
# ルール生成は、ルールを作成した時点の辞書（v1）とその検証で行った。
from .normalize_v1 import (CANCER_ALIASES, CANCER_PATTERNS, RELATIONSHIP_PATTERNS, SemanticOutputError,
                           StructuredOutputError, parse_strict_response, strict_pairs, validate_semantic_response)
from .rule_engine import predict, prepare_rule_input, rule_schema, validate_rules


MODEL_ID = "qwen3.5-27b-q4_k_m"
N_CTX = 65536
MAX_TOKENS = 32768
MAXIMUM_RULES = 100
MINIMUM_EXAMPLES_PER_LIST = 4
REGEX_TIMEOUT_SECONDS = 1.0
MODEL_SETTINGS = {"n_threads": 12, "n_gpu_layers": -1, "n_batch": 256, "n_ubatch": 128, "seed": 20261020}
SLOTS = {"$relation": r"\g<relation>", "$disease": r"\g<disease>"}


# ---- 疑似正解と例の選択 ----------------------------------------------------------------

def development_records(candidate_dir: Path, patient_ids: set[str],
                        response_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Qwen3.5のFSの抽出結果を疑似正解とする。検証に失敗した出力と、入力を切り捨てた出力は除く。

    検証は辞書v1による内容の検証（normalize_v1.validate_semantic_response）で行う。
    """

    records, excluded = [], []
    responses: dict[str, dict[str, Any]] = {}
    for record in candidate_records(candidate_dir, patient_ids):
        patient_id = record["patient_id"]
        if patient_id not in responses:
            path = response_dir / f"{patient_id}_withResponse.json"
            responses[patient_id] = load_json(path) if path.is_file() else {}
        item = responses[patient_id].get(record["date"])
        try:
            if not isinstance(item, dict) or item.get("status") not in ("success", "semantic_invalid"):
                raise ValueError("出力がないか、出力の検証に失敗しています")
            meta = item.get("_meta", {})
            if meta.get("model_id") != MODEL_ID or meta.get("condition") != "few_shot":
                raise ValueError("Qwen3.5のFSの出力が必要です")
            if meta.get("source_truncated") is not False:
                raise ValueError("入力が切り捨てられています")
            if meta.get("source_sha256") != record["source_sha256"]:
                raise ValueError("出力と候補テキストの原文が一致しません")
            prediction = (parse_strict_response(item["raw_response"]) if "raw_response" in item
                          else item["prediction"])
            try:
                validate_semantic_response(prediction, record["source_text"].strip())
            except (StructuredOutputError, SemanticOutputError) as exc:
                raise ValueError("出力がないか、出力の検証に失敗しています") from exc
            records.append({**record, "label_pairs": strict_pairs(prediction)})
        except (ValueError, KeyError, TypeError) as exc:
            excluded.append({**record, "reason": str(exc)})
    return records, excluded


def row_payload(row: dict[str, Any]) -> dict[str, Any]:
    """プロンプトに入れる1件分の例。原文と疑似正解のペアに加え、作り直しでは初期ルールの出力も含める。"""

    result = {"source_text": prepare_rule_input(row["source_text"]), "expected_pairs": row["label_pairs"]}
    for key in ("predicted_pairs", "execution_error"):
        if key in row:
            result[key] = row[key]
    return result


def representative_examples(records: list[dict[str, Any]], maximum: int, byte_budget: int) -> list[dict[str, Any]]:
    """まだ含まれていないペアを多く含む例を優先し、同点なら短い原文を選ぶ。原文は切り詰めない。"""

    remaining = list(records)
    selected, seen, covered = [], set(), set()
    used = 0
    while remaining and len(selected) < maximum:
        def key(row: dict[str, Any]) -> tuple[int, int, str]:
            gain = len(set(map(tuple, row.get("label_pairs", []))) - covered)
            return (-gain, len(row["source_text"].encode()), json_sha256([row["source_sha256"], row.get("label_pairs")]))
        row = min(remaining, key=key)
        remaining.remove(row)
        size = len(json.dumps(row_payload(row), ensure_ascii=False).encode())
        if row["source_sha256"] in seen or used + size > byte_budget:
            continue
        selected.append(row)
        seen.add(row["source_sha256"])
        covered.update(map(tuple, row.get("label_pairs", [])))
        used += size
    return selected


def initial_examples(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """家族歴ありの例と、がんなどの病名を含む家族歴なしの例を選ぶ。"""

    positives = representative_examples([r for r in records if r["label_pairs"]], 48, 72000)
    informative = [r for r in records if not r["label_pairs"]
                   and re.search(r"癌|がん|ガン|腫瘍|家族歴|既往歴", r["source_text"])]
    negatives = representative_examples(informative, 24, 24000)
    return {"positive": positives, "negative": negatives}


# ---- 指示文とJSON Schema -----------------------------------------------------------------

def instruction(mode: str = "initial") -> str:
    reference = {"pattern": "(?P<relation>父|母)[：:は]?(?P<disease>胃癌|肺癌)",
                 "relationships": ["$relation"], "cancers": ["$disease"]}
    return (
        "日本語カルテ原文から血縁者の続柄–がん種ペアを抽出する、単独実行可能な正規表現ルールを生成してください。"
        "教師は原文とQwen疑似抽出結果です。疑似正解は誤りを含み得るので、定義と原文を優先し、"
        "誤った疑似正解へ合わせるためのルールは作らないでください。"
        "血縁者の肯定された対象がんのみ採用します。本人、非血縁者、否定、疑い、主体不明、"
        "家族への説明・連絡・同席だけの記載は除外します。別人物・別疾患へ否定や疾患を引き継がないでください。"
        "各ruleはid,pattern,priority,reason,relationships,cancers,emit_pairを持ちます。"
        "全patternで疾患語を(?P<disease>...)に捕捉し、続柄を原文から取る場合は(?P<relation>...)に捕捉してください。"
        "disease groupは原文に実際に書かれる表記（例: 胃癌、胃がん、早期胃癌、結腸癌、直腸癌、乳癌、膵癌）に一致させてください。"
        "対象がん分類（胃がん、大腸がん等）は正規化後のラベルであり、原文の語彙ではありません。"
        "cancersに\"$disease\"を使うと捕捉した表記が実行系で分類へ変換されますが、変換できない表記"
        "（例: 胃ガン、肺腺癌）はそのルールのcancersに固定ラベル（例: \"胃がん\"、\"肺がん\"）を指定してください。"
        "relationshipsは\"$relation\"（patternのrelation groupの一致文字列）か、対象続柄の固定値です。"
        "cancersは\"$disease\"（patternのdisease groupの一致文字列）か、対象がん分類の固定値です。"
        "それ以外の文字列やg<relation>、\\g<relation>は使えません。例: " + json.dumps(reference, ensure_ascii=False) + "。"
        "語彙は網羅してください。21がん分類それぞれについて、癌・がん・ガンの表記、臓器名の別表記"
        "（例: 膵癌・膵臓癌・膵がん、腎癌・腎がん・腎臓癌、肝癌・肝臓癌、食道癌、子宮体癌・子宮頸癌、胆管癌・胆嚢癌）、"
        "組織型や部位を含む表記（例: 肺腺癌、S状結腸癌）を、本文例に現れない分類も含めて用意してください。"
        "続柄も別表記（例: 父親、母親、実父、実母、おじ、おば）を含めてください。兄弟・姉妹・両親・祖父母のように1人に決まらない語はrelationに捕捉しないでください。"
        "実行系が正規化に使う辞書を示します。patternの語彙はこれらの表記を網羅し、本文例にない表記も含めてください。"
        "続柄の表記（ラベル: 正規表現）=" + json.dumps(RELATIONSHIP_PATTERNS, ensure_ascii=False) +
        "。がんの表記（分類: 正規表現）=" + json.dumps(CANCER_PATTERNS, ensure_ascii=False) +
        "。がんの別名（表記: 分類）=" + json.dumps(CANCER_ALIASES, ensure_ascii=False) + "。"
        "これらの表記に「ガン」や接頭語（早期、進行、再発、術後など）が付く場合も考慮してください。"
        "区切りは全角・半角のコロン、セミコロン（;；）、読点、カンマ、空白、括弧、改行を扱ってください。"
        "同じ人物の疾患の列挙（例: 父:胃癌、食道癌）は各疾患をその人物のペアとして抽出してください。"
        "1つの一致からは続柄とがん種の全組合せが出力されるため、別人物の疾患を同じルールに混ぜないでください。"
        "re.VERBOSEは無効です。(父 | 母)ではなく(父|母)と書き、選択肢の前後に空白を入れないでください。"
        "全角・半角の文字は別の文字です。どの選択肢でもがん名の一致を必須にしてください。"
        "Python reとregex VERSION0で全一致を検索します。疾患groupの範囲が重なればpriority降順・同点JSON順で1つを採用します。"
        "emit_pair=falseが採用されるとその疾患位置の陽性を抑制します。別人物の否定で文全体を除外しないでください。"
        "無制限の.*、入れ子量指定、可変長後読み、長文を何度も走査する複雑なpatternを避け、"
        "人物–疾患の間は短い区切りと助詞などで限定してください。語彙や構文は共通化し重複を避けます。"
        f"ルールは1～{MAXIMUM_RULES}本です。{MAXIMUM_RULES}本は上限であり目標ではありません。"
        "必要な構文・例外を網羅し、数を満たすための水増しはしないでください。"
        "id・priority・reasonだけが異なる同じpatternは重複であり禁止です。"
        "否定・疑い・本人・非血縁者・説明相手などを除外するemit_pair=falseのルールを、"
        "対応する陽性ルールより高いpriorityで用意してください。"
        "emit_pair=falseのルールの出力欄は使われませんが、patternに(?P<relation>...)がなければ"
        "relationshipsに\"$relation\"ではなく固定の続柄（例: \"父\"）を書いてください。"
        "書式に不備のあるルールは実行時に除外されます。"
        "抽出できない場合は空ペア集合になります。有無はペアが1件以上かで計算します。"
        "対象続柄=" + json.dumps(list(BLOOD_RELATIONSHIPS), ensure_ascii=False) +
        "。対象がん分類（cancersの固定ラベル。patternの語彙ではない）=" + json.dumps(list(CANCER_TYPES), ensure_ascii=False) + "。"
        "descriptionには方針と限界を記してください。"
        + TASK_TEXT[mode]
    )


TASK_TEXT = {
    "initial": "以下の原文と期待ペアから初期ルールの完全な集合を生成してください。",
    "revision": "既存ルール、誤分類原文（expected_pairsが期待値、predicted_pairsが既存ルールの実出力）、"
                "正解維持用の例から、修正版の完全なルール集合を1回で生成してください。"
                "省略したルールは削除扱いです。誤分類例だけに過剰適合せず、正しかった例と除外条件を保ってください。",
}


def typed_schema() -> dict[str, Any]:
    """生成させるルール一式のJSON Schema。続柄とがん種は、所定の名称か$relation・$diseaseに限る。"""

    schema = rule_schema()
    rule = schema["properties"]["rules"]["items"]
    rule["properties"]["relationships"] = {"type": "array", "minItems": 1,
        "items": {"type": "string", "enum": list(BLOOD_RELATIONSHIPS) + ["$relation"]}}
    rule["properties"]["cancers"] = {"type": "array", "minItems": 1,
        "items": {"type": "string", "enum": list(CANCER_TYPES) + ["$disease"]}}
    schema["properties"]["rules"]["maxItems"] = MAXIMUM_RULES
    return schema


def compile_typed(spec: dict[str, Any]) -> dict[str, Any]:
    """$relation・$diseaseをgroup参照へ置き換え、実行できるかを確かめる。patternは変えない。"""

    if not 1 <= len(spec["rules"]) <= MAXIMUM_RULES:
        raise ValueError(f"ルール数は1～{MAXIMUM_RULES}本が必要です")
    result = copy.deepcopy(spec)
    for rule in result["rules"]:
        if re.search(r"(?<!\\)\s\||\|\s", rule["pattern"]):
            raise ValueError(f"{rule['id']}: |の前後に空白があります")
        pattern = re.compile(rule["pattern"])
        for field, slot, allowed in [("relationships", "$relation", BLOOD_RELATIONSHIPS),
                                     ("cancers", "$disease", CANCER_TYPES)]:
            converted = []
            for value in rule[field]:
                if value == slot:
                    if slot[1:] not in pattern.groupindex:
                        raise ValueError(f"{rule['id']}: {slot}には(?P<{slot[1:]}>...)が必要です")
                    converted.append(SLOTS[slot])
                elif value in allowed:
                    converted.append(value)
                else:
                    raise ValueError(f"{rule['id']}: {field}の値が不正です: {value!r}")
            rule[field] = converted
    if len({r["id"] for r in result["rules"]}) != len(result["rules"]):
        raise ValueError("ルールidが重複しています")
    patterns = [r["pattern"] for r in result["rules"]]
    if len(set(patterns)) != len(patterns):
        raise ValueError("同じpatternのルールが重複しています")
    validate_rules(result)
    return result


def typed_view(spec: dict[str, Any]) -> dict[str, Any]:
    """実行形式のルールを、Qwen3.5が返す形式（$relation・$disease）に戻して示す。"""

    result = copy.deepcopy(spec)
    reverse = {value: key for key, value in SLOTS.items()}
    for rule in result["rules"]:
        for field in ("relationships", "cancers"):
            rule[field] = [reverse.get(value, value) for value in rule[field]]
    return result


# ---- 生成 ----------------------------------------------------------------------------------

def prompt_tokens(model: Any, messages: list[dict[str, str]]) -> int:
    prompt = model._family_history_chat_formatter(messages=messages).prompt
    return len(model.tokenize(prompt.encode(), add_bos=False, special=True))


def fitted_messages(model: Any, build: Callable[..., list[dict[str, str]]],
                    *ranked: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """プロンプト全体がコンテキストに収まるまで、最も多い例の一覧から優先度の低い例を除く。"""

    budget = N_CTX - MAX_TOKENS - 128
    lists = [list(items) for items in ranked]
    history = []
    while True:
        messages = build(*lists)
        tokens = prompt_tokens(model, messages)
        history.append({"counts": [len(items) for items in lists], "prompt_tokens": tokens})
        if tokens <= budget:
            return messages, lists, history
        longest = max(range(len(lists)), key=lambda i: len(lists[i]))
        if len(lists[longest]) <= MINIMUM_EXAMPLES_PER_LIST:
            raise ValueError(f"最小の例の数でもコンテキストに収まりません: {history[-1]}")
        lists[longest].pop()


def call_model(model: Any, messages: list[dict[str, str]], schema: dict[str, Any], directory: Path) -> dict[str, Any]:
    """構造化出力でルール一式を1回生成させ、入力・出力・所要時間を保存する。"""

    if model._family_history_native_thinking_control != "disabled_enable_thinking_false":
        raise ValueError("thinkingモードを無効にできていません")
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_json(directory / "prompt_messages.json", messages)
    write_json(directory / "schema.json", schema)
    meta: dict[str, Any] = {"max_tokens": MAX_TOKENS, "n_ctx": N_CTX}
    started = perf_counter()
    try:
        meta["prompt_tokens"] = prompt_tokens(model, messages)
        if meta["prompt_tokens"] + MAX_TOKENS + 128 > N_CTX:
            raise ValueError("コンテキストが不足しています")
        response = model.create_chat_completion(messages=messages, max_tokens=MAX_TOKENS, **GENERATION_CONFIG,
                                                response_format={"type": "json_object", "schema": schema})
        write_json(directory / "generation_response.json", response)
        meta["usage"] = usage_of(response)
        if response["choices"][0].get("finish_reason") != "stop":
            raise ValueError("出力が上限で打ち切られました")
        document = json.loads(extract_content(response))
        write_json(directory / "document.json", document)
        return document
    finally:
        meta["elapsed_seconds"] = perf_counter() - started
        write_json(directory / "generation_metadata.json", meta)


def keep_executable(document: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """生成されたルールのうち、書かれたとおりに実行できるものだけを残す。"""

    kept, dropped, ids, patterns = [], [], set(), set()
    for rule in document["rules"]:
        try:
            if rule["id"] in ids or rule["pattern"] in patterns:
                raise ValueError("idまたはpatternが前のルールと重複しています")
            compile_typed({**document, "rules": [rule]})
        except (ValueError, re.error, KeyError, TypeError) as exc:
            dropped.append({"id": rule.get("id"), "error": str(exc)})
            continue
        ids.add(rule["id"])
        patterns.add(rule["pattern"])
        kept.append(rule)
    if not kept:
        raise ValueError("実行できるルールがありません")
    return compile_typed({**document, "rules": kept}), kept, dropped


def generate_rules(model: Any, messages: list[dict[str, str]], directory: Path) -> dict[str, Any]:
    document = call_model(model, messages, typed_schema(), directory / "call_00")
    spec, kept, dropped = keep_executable(document)
    write_json(directory / "technical_validation.json", {
        "generated_rules": len(document["rules"]), "kept_rules": len(kept), "dropped_rules": dropped})
    write_json(directory / "rules.json", spec)
    return spec


# ---- 手順 ----------------------------------------------------------------------------------

def prepare(candidate_dir: Path, development_list: Path, response_dir: Path, output_root: Path) -> None:
    """疑似正解を作り、初期ルールの例を選ぶ。"""

    if output_root.exists():
        raise FileExistsError(output_root)
    records, excluded = development_records(candidate_dir, load_patient_ids(development_list), response_dir)
    examples = initial_examples(records)
    if not examples["positive"] or not examples["negative"]:
        raise ValueError("例を選べません")
    write_json(output_root / "development_records.json", records)
    write_json(output_root / "excluded_records.json", excluded)
    write_json(output_root / "initial_examples.json", examples)


def errors_and_correct(records: list[dict[str, Any]], predictions: list[dict[str, Any]]):
    errors, correct = [], []
    for record, prediction in zip(records, predictions):
        if prediction["status"] != "success" or record["label_pairs"] != prediction["pairs"]:
            errors.append({**record, "predicted_pairs": prediction["pairs"],
                           "execution_error": prediction.get("error")})
        else:
            correct.append(record)
    return errors, correct


def develop(output_root: Path, model_path: Path, model_factory: Callable[[ModelSettings], Any] = load_model) -> None:
    """初期ルールを生成し、誤分類例を与えて1回だけ作り直させる。作り直したルールを最終版とする。"""

    records = load_json(output_root / "development_records.json")
    examples = load_json(output_root / "initial_examples.json")
    model = model_factory(ModelSettings(model_path=model_path, model_id=MODEL_ID, n_ctx=N_CTX,
                                        flash_attention=True, **MODEL_SETTINGS))
    try:
        def initial_messages(positives, negatives):
            return [{"role": "system", "content": instruction("initial")},
                    {"role": "user", "content": json.dumps(
                        {"examples": [row_payload(r) for r in positives + negatives]}, ensure_ascii=False)}]
        messages, (positives, negatives), fit = fitted_messages(
            model, initial_messages, examples["positive"], examples["negative"])
        write_json(output_root / "initial_prompt_fit.json",
                   {"history": fit, "positive": len(positives), "negative": len(negatives)})
        initial = generate_rules(model, messages, output_root / "initial")

        predictions = predict(initial, records, REGEX_TIMEOUT_SECONDS)
        write_json(output_root / "initial" / "development_predictions.json", predictions)
        errors, correct = errors_and_correct(records, predictions)
        ranked_errors = representative_examples(errors, 48, 72000)
        ranked_anchors = representative_examples(correct, 12, 12000)
        if not ranked_errors:
            raise ValueError("誤分類例がありません")

        def revision_messages(selected_errors, anchors):
            return [{"role": "system", "content": instruction("revision")},
                    {"role": "user", "content": json.dumps({
                        "existing_rules": typed_view(initial),
                        "misclassified_examples": [row_payload(r) for r in selected_errors],
                        "correct_anchors": [row_payload(r) for r in anchors]}, ensure_ascii=False)}]
        messages, (selected_errors, anchors), fit = fitted_messages(
            model, revision_messages, ranked_errors, ranked_anchors)
        write_json(output_root / "regeneration_examples.json",
                   {"errors": selected_errors, "correct_anchors": anchors, "prompt_fit": fit})
        revised = generate_rules(model, messages, output_root / "revised")
    finally:
        if hasattr(model, "close"):
            model.close()
    write_json(output_root / "revised" / "development_predictions.json",
               predict(revised, records, REGEX_TIMEOUT_SECONDS))
    write_json(output_root / "final_rules.json", revised)


def apply_rules(rules_path: Path, candidate_dir: Path, patient_list: Path, output_file: Path) -> None:
    """最終版のルールを評価用のテキストに適用する。LLMは使わない。"""

    records = candidate_records(candidate_dir, load_patient_ids(patient_list))
    write_json(output_file, predict(load_json(rules_path), records, REGEX_TIMEOUT_SECONDS))
