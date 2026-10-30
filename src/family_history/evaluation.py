"""人手正解との照合による評価。

判定不能のテキストを除いて、家族歴有無の正解率・適合率・再現率・F1値、ペア集合の正解率、
ペア単位の適合率・再現率・F1値、家族歴ありのテキストに限ったペア集合の正解率を求める。
LLMの出力は形式だけを確認し、JSONとして解析できないかSchemaに適合しない出力（出力形式エラー）と
ルールの適用エラーは、ペアを抽出しなかった出力として数え、正解率ではすべて不正解とする。
それ以外の出力は、「家族歴として採用」がtrueの項目をそのまま採点し、正規化できない採用項目は
誤って抽出されたペアとして数える。
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from .io_utils import load_json
from .normalize import adopted_pairs, family_history_pairs, parse_strict_response

Key = tuple[str, str]


def read_reference(path: Path) -> dict[Key, set[tuple[str, str]]]:
    """人手正解CSV（patient_id、date、decision、gold_family_history_json）を読む。判定不能は除く。"""

    reference = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["decision"] == "undecidable":
                continue
            reference[(row["patient_id"], row["date"])] = family_history_pairs(
                json.loads(row["gold_family_history_json"]))
    return reference


def llm_predictions(*response_dirs: Path) -> dict[Key, dict[str, Any]]:
    """1つのモデル・プロンプト条件の出力を読む。複数のディレクトリに分けて推論した場合はまとめて読む。"""

    predictions = {}
    for response_dir in response_dirs:
        for path in sorted(Path(response_dir).glob("*_withResponse.json")):
            patient_id = path.name[: -len("_withResponse.json")]
            for date, item in load_json(path).items():
                key = (patient_id, date)
                if key in predictions:
                    raise ValueError(f"同じテキストの出力が重複しています: {key}")
                success = item["status"] != "json_invalid"
                pairs: set[tuple[str, str]] = set()
                if success:
                    document = (parse_strict_response(item["raw_response"]) if "raw_response" in item
                                else item["prediction"])
                    pairs = adopted_pairs(document)
                predictions[key] = {
                    "success": success,
                    "pairs": pairs,
                    "latency_sec": item["_meta"]["latency_sec"],
                    "source_truncated": item["_meta"]["source_truncated"],
                }
    return predictions


def rule_predictions(path: Path) -> dict[Key, dict[str, Any]]:
    """ルールベース手法の予測JSONを読む。"""

    return {(row["patient_id"], row["date"]): {
        "success": row["status"] == "success",
        "pairs": {tuple(pair) for pair in row["pairs"] or []},
        "latency_sec": row["elapsed_seconds"],
    } for row in load_json(path)}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metrics(reference: dict[Key, set], predictions: dict[Key, dict[str, Any]]) -> dict[str, Any]:
    missing = sorted(set(reference) - set(predictions))
    if missing:
        raise ValueError(f"予測がないテキストがあります: {missing[:3]}")
    n = len(reference)
    presence = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    tp = fp = fn = exact = failures = positive = positive_exact = 0
    for key in sorted(reference):
        gold = reference[key]
        prediction = predictions[key]
        predicted = prediction["pairs"] if prediction["success"] else set()
        failures += not prediction["success"]
        if prediction["success"]:
            cell = ("tp" if predicted else "fn") if gold else ("fp" if predicted else "tn")
            presence[cell] += 1
        elif gold:
            presence["fn"] += 1
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
        correct = prediction["success"] and predicted == gold
        exact += correct
        if gold:
            positive += 1
            positive_exact += correct
    p_tp, p_fp, p_fn = presence["tp"], presence["fp"], presence["fn"]
    return {
        "evaluated_texts": n,
        "json_output_errors_or_rule_errors": failures,
        "presence_accuracy": (p_tp + presence["tn"]) / n,
        "presence_precision": _ratio(p_tp, p_tp + p_fp),
        "presence_recall": _ratio(p_tp, p_tp + p_fn),
        "presence_f1": _ratio(2 * p_tp, 2 * p_tp + p_fp + p_fn),
        "pair_set_accuracy": exact / n,
        "pair_precision": _ratio(tp, tp + fp),
        "pair_recall": _ratio(tp, tp + fn),
        "pair_f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "true_positive_pairs": tp, "false_positive_pairs": fp, "false_negative_pairs": fn,
        "texts_with_family_history": positive,
        "pair_set_accuracy_in_texts_with_family_history": _ratio(positive_exact, positive),
    }


def inference_time(predictions: dict[Key, dict[str, Any]]) -> dict[str, float]:
    """全テキスト（判定不能を含む）の処理時間の合計・平均・中央値。"""

    values = [prediction["latency_sec"] for prediction in predictions.values()]
    return {"texts": len(values), "total_seconds": sum(values), "total_hours": sum(values) / 3600,
            "mean_seconds": statistics.mean(values), "median_seconds": statistics.median(values)}
