"""コマンドラインの入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import evaluation, rule_generation
from .constants import CONDITIONS
from .inference import ModelSettings, run_inference
from .io_utils import candidate_records, load_patient_ids, write_json
from .selection import select_development_patients, select_evaluation_patients
from .simple_rules import apply_simple_rules


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="family_history", description="日本語電子カルテからのがん家族歴抽出")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("select-evaluation", help="評価用の患者を無作為に抽出する")
    p.add_argument("--candidate-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = commands.add_parser("select-development", help="評価用と重ならないルール生成用の患者を抽出する")
    p.add_argument("--candidate-dir", type=Path, required=True)
    p.add_argument("--evaluation-list", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = commands.add_parser("infer", help="LLMで抽出する")
    p.add_argument("--candidate-dir", type=Path, required=True)
    p.add_argument("--patient-list", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--n-ctx", type=int, required=True)
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--output-root", type=Path, required=True)

    p = commands.add_parser("simple-rules", help="単純ルールで抽出する")
    p.add_argument("--candidate-dir", type=Path, required=True)
    p.add_argument("--patient-list", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = commands.add_parser("prepare-rule-generation", help="疑似正解を作り、ルール生成の例を選ぶ")
    p.add_argument("--candidate-dir", type=Path, required=True)
    p.add_argument("--development-list", type=Path, required=True)
    p.add_argument("--response-dir", type=Path, required=True, help="ルール生成用の患者に対するQwen3.5のFSの出力")
    p.add_argument("--output-root", type=Path, required=True)

    p = commands.add_parser("generate-rules", help="Qwen3.5に初期ルールを生成させ、1回作り直させる")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)

    p = commands.add_parser("apply-rules", help="生成したルールを適用する")
    p.add_argument("--rules", type=Path, required=True)
    p.add_argument("--candidate-dir", type=Path, required=True)
    p.add_argument("--patient-list", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)

    p = commands.add_parser("evaluate", help="人手正解に対して評価する")
    p.add_argument("--reference-csv", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--llm-responses", type=Path, nargs="+",
                        help="1つのモデル・条件のresponsesディレクトリ（分けて推論した場合は複数）")
    source.add_argument("--rule-predictions", type=Path, help="ルールベース手法の予測JSON")
    p.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "select-evaluation":
        select_evaluation_patients(args.candidate_dir, args.output)
    elif args.command == "select-development":
        select_development_patients(args.candidate_dir, load_patient_ids(args.evaluation_list), args.output)
    elif args.command == "infer":
        run_inference(args.candidate_dir, args.patient_list, args.output_root,
                      ModelSettings(model_path=args.model_path, model_id=args.model_id, n_ctx=args.n_ctx),
                      conditions=tuple(args.conditions))
    elif args.command == "simple-rules":
        records = candidate_records(args.candidate_dir, load_patient_ids(args.patient_list))
        write_json(args.output, apply_simple_rules(records))
    elif args.command == "prepare-rule-generation":
        rule_generation.prepare(args.candidate_dir, args.development_list, args.response_dir, args.output_root)
    elif args.command == "generate-rules":
        rule_generation.develop(args.output_root, args.model_path)
    elif args.command == "apply-rules":
        rule_generation.apply_rules(args.rules, args.candidate_dir, args.patient_list, args.output)
    elif args.command == "evaluate":
        reference = evaluation.read_reference(args.reference_csv)
        if args.llm_responses:
            predictions = evaluation.llm_predictions(*args.llm_responses)
        else:
            predictions = evaluation.rule_predictions(args.rule_predictions)
        result = {**evaluation.metrics(reference, predictions),
                  "processing_time": evaluation.inference_time(predictions)}
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
