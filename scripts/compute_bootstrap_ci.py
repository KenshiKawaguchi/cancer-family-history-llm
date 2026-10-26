#!/usr/bin/env python3
"""患者単位のクラスター・ブートストラップ法で、各指標と条件間の差の95%信頼区間を求める。

評価対象のテキストを含む患者を復元抽出し、抽出された患者のテキストをすべて用いて指標を再計算して、
パーセンタイル法で区間を定める。2条件の差は、同じ再標本で両方の条件を計算して求める。
誤りまたは正解が0件で区間の幅が0となる指標は、該当する患者数を件数とみなしたクロッパー・ピアソン法で求める。
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from family_history import evaluation  # noqa: E402

MODELS = {"qwen3.5-27b-q4_k_m": "Qwen3.5", "gemma-3-27b-it-q4_k_m": "Gemma 3",
          "medgemma-27b-text-it-q4_k_m": "MedGemma", "llm-jp-3.1-13b-instruct4-q8_0": "LLM-jp",
          "sip-jmed-llm-3-13b-op-32k-r0.1-q8_0": "SIP-jmed"}
CONDITIONS = {"zero_shot": "ZS", "few_shot": "FS", "zero_shot_cot": "ZS+C", "few_shot_cot": "FS+C"}
FIELDS = ["n", "exact", "pres", "tp", "fp", "fn", "npos", "posexact", "ptp", "pfp", "pfn", "pred_pos"]

# 本文で比較した条件の組（前者 - 後者）と指標。
COMPARISONS = (
    [((m, "FS"), (m, "ZS"), "presence_accuracy") for m in MODELS.values()]
    + [((m, "FS+C"), (m, "FS"), "presence_accuracy") for m in MODELS.values()]
    + [((m, "FS+C"), (m, "FS"), "pair_f1") for m in ("MedGemma", "LLM-jp", "SIP-jmed")]
    + [(("SIP-jmed", "ZS+C"), ("SIP-jmed", "ZS"), m) for m in ("presence_accuracy", "pair_recall")]
    + [(("MedGemma", c), ("Gemma 3", c), "presence_accuracy") for c in CONDITIONS.values()]
    + [(("MedGemma", "FS"), ("Gemma 3", "FS"), m) for m in ("pair_recall", "pair_f1", "positive_pairset_accuracy")]
    + [(("LLM-jp", "FS"), ("SIP-jmed", "FS"), m) for m in ("presence_accuracy", "pair_f1")]
    + [(("LLM-jp", "ZS+C"), ("SIP-jmed", "ZS+C"), "presence_accuracy")]
    + [(("rule", "qwen"), ("rule", "simple"), m)
       for m in ("presence_f1", "pair_precision", "pair_recall", "pair_f1", "positive_pairset_accuracy")]
    + [(("Qwen3.5", "FS"), ("rule", "qwen"), m) for m in ("pair_recall", "pair_f1", "positive_pairset_accuracy")]
)


def per_patient(reference, predictions, patient_index):
    """テキストごとの結果を、患者ごとの件数に集計する。"""

    totals = {key: np.zeros(len(patient_index)) for key in FIELDS}
    for (patient_id, date), gold in reference.items():
        prediction = predictions[(patient_id, date)]
        success = prediction["success"]
        predicted = prediction["pairs"] if success else set()
        i = patient_index[patient_id]
        values = {"n": 1, "exact": int(success and predicted == gold),
                  "pres": int(success and bool(predicted) == bool(gold)),
                  "tp": len(gold & predicted), "fp": len(predicted - gold), "fn": len(gold - predicted),
                  "npos": int(bool(gold)), "posexact": int(bool(gold) and success and predicted == gold),
                  "ptp": int(bool(gold) and bool(predicted)), "pfp": int(not gold and bool(predicted)),
                  "pfn": int(bool(gold) and not predicted), "pred_pos": int(bool(predicted))}
        for key, value in values.items():
            totals[key][i] += value
    return totals


def metrics(sums):
    def div(a, b):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)
    tp, fp, fn = sums["tp"], sums["fp"], sums["fn"]
    ptp, pfp, pfn = sums["ptp"], sums["pfp"], sums["pfn"]
    return {"presence_accuracy": div(sums["pres"], sums["n"]), "pairset_accuracy": div(sums["exact"], sums["n"]),
            "positive_pairset_accuracy": div(sums["posexact"], sums["npos"]),
            "pair_precision": div(tp, tp + fp), "pair_recall": div(tp, tp + fn), "pair_f1": div(2 * tp, 2 * tp + fp + fn),
            "presence_precision": div(ptp, ptp + pfp), "presence_recall": div(ptp, ptp + pfn),
            "presence_f1": div(2 * ptp, 2 * ptp + pfp + pfn)}


def clopper_pearson(successes, n, alpha=0.05):
    """全例が成功または全例が失敗の場合のクロッパー・ピアソン区間。"""
    if successes == n:
        return (alpha / 2) ** (1 / n), 1.0
    return 0.0, 1 - (alpha / 2) ** (1 / n)


def interval(point, boot, totals, metric):
    value = point[metric]
    if math.isnan(value):
        return value, None, None, "undefined"
    samples = boot[metric][~np.isnan(boot[metric])]
    if np.ptp(samples) > 0:
        low, high = np.percentile(samples, [2.5, 97.5])
        return value, float(low), float(high), "bootstrap"
    if metric == "pair_precision" and value == 1.0:
        n = int(((totals["tp"] + totals["fp"]) > 0).sum())
        return (value, *clopper_pearson(n, n), f"clopper_pearson_patients={n}")
    if metric == "presence_precision" and value == 1.0:
        n = int((totals["pred_pos"] > 0).sum())
        return (value, *clopper_pearson(n, n), f"clopper_pearson_patients={n}")
    if metric in ("pair_recall", "presence_recall", "positive_pairset_accuracy") and value == 0.0:
        n = int((totals["npos"] > 0).sum())
        return (value, *clopper_pearson(0, n), f"clopper_pearson_patients={n}")
    return value, None, None, "not_calculated"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, required=True, help="人手正解CSV")
    parser.add_argument("--llm-results", type=Path, required=True,
                        help="<モデルID>/<条件>/responses を含むLLMの出力ディレクトリ")
    parser.add_argument("--simple-rule-predictions", type=Path, required=True)
    parser.add_argument("--qwen-rule-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"既存の出力を上書きしません: {args.output}")

    reference = evaluation.read_reference(args.reference_csv)
    patients = sorted({patient_id for patient_id, _ in reference})
    patient_index = {patient_id: i for i, patient_id in enumerate(patients)}
    systems = {}
    for responses in sorted(args.llm_results.glob("**/responses")):
        model, condition = responses.parts[-3], responses.parts[-2]
        if model in MODELS and condition in CONDITIONS:
            systems[(MODELS[model], CONDITIONS[condition])] = evaluation.llm_predictions(responses)
    systems[("rule", "simple")] = evaluation.rule_predictions(args.simple_rule_predictions)
    systems[("rule", "qwen")] = evaluation.rule_predictions(args.qwen_rule_predictions)

    rng = np.random.default_rng(args.seed)
    weights = np.stack([np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))
                        for _ in range(args.n_boot)])
    totals = {key: per_patient(reference, predictions, patient_index) for key, predictions in systems.items()}
    point = {key: {m: float(v) for m, v in metrics({k: a.sum() for k, a in t.items()}).items()}
             for key, t in totals.items()}
    boot = {key: metrics({k: weights @ a for k, a in t.items()}) for key, t in totals.items()}

    differences = []
    for first, second, metric in COMPARISONS:
        samples = boot[first][metric] - boot[second][metric]
        samples = samples[~np.isnan(samples)]
        low, high = np.percentile(samples, [2.5, 97.5])
        differences.append({"first": list(first), "second": list(second), "metric": metric,
                            "difference": point[first][metric] - point[second][metric],
                            "low": float(low), "high": float(high)})
    result = {"n_boot": args.n_boot, "seed": args.seed, "patients": len(patients), "texts": len(reference),
              "estimates": {f"{k[0]}|{k[1]}": {m: interval(point[k], boot[k], totals[k], m) for m in point[k]}
                            for k in totals},
              "differences": differences}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"patients {len(patients)}, texts {len(reference)} -> {args.output}")


if __name__ == "__main__":
    main()
