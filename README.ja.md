# 日本語電子カルテからのがん家族歴抽出

[English](README.md)

論文「Performance and Error Analysis of Large Language Models for Extracting Cancer Family History From Japanese Electronic Health Records of Patients With Colorectal Cancer」（大腸がん患者の日本語電子カルテからのがん家族歴抽出における大規模言語モデルの性能と誤りの分析）で用いたコードです。

患者データ、モデルの重み、実験結果は含みません。診療データは公開できません。コード中のコメントとメッセージは日本語で書かれています。

## 内容

| パス | 役割 |
|---|---|
| `src/family_history/selection.py` | 評価用とルール生成用の患者の抽出 |
| `src/family_history/prompts.py` | プロンプト |
| `src/family_history/inference.py` | LLMによる抽出 |
| `src/family_history/normalize.py` | 出力の検証と正規化 |
| `src/family_history/simple_rules.py` | 単純ルール |
| `src/family_history/rule_generation.py` | Qwen3.5によるルール生成 |
| `src/family_history/rule_engine.py` | ルールの検証と適用 |
| `src/family_history/evaluation.py` | 評価 |
| `scripts/compute_bootstrap_ci.py` | 95%信頼区間の計算 |
| `scripts/summarize_age_sex.py` | 性別と年齢の集計 |

## 入力

- 候補テキスト：患者ごとのJSONファイル（`<患者ID>.json`）。日付をキーとし、値は`{"fam_text": 本文}`です。キーワード検索で取り出した経過記録の本文を入れます。
- 人手正解：`patient_id`、`date`、`decision`（`family_history`、`no_history`、`undecidable`）、`gold_family_history_json`の列を持つCSVです。

## 使い方

このディレクトリで、`PYTHONPATH=src`を付けて実行します。各コマンドの引数は`--help`で確認できます。

```
PYTHONPATH=src python -m family_history select-evaluation       # 評価用の患者を抽出
PYTHONPATH=src python -m family_history infer                   # LLMによる抽出（4つのプロンプト条件）
PYTHONPATH=src python -m family_history simple-rules            # 単純ルールによる抽出
PYTHONPATH=src python -m family_history select-development      # ルール生成用の患者を抽出
PYTHONPATH=src python -m family_history prepare-rule-generation # 疑似正解の作成と例の選択
PYTHONPATH=src python -m family_history generate-rules          # ルールの生成と1回の作り直し
PYTHONPATH=src python -m family_history apply-rules             # 生成したルールの適用
PYTHONPATH=src python -m family_history evaluate                # 人手正解に対する評価
python scripts/compute_bootstrap_ci.py                          # 95%信頼区間
```

`infer`のコンテキスト長（`--n-ctx`）は、LLM-jpでは4096、ほかのモデルでは8192としました。

LLMの推論とルール生成には`llama-cpp-python`とCUDA対応のGPUが、ルールの適用には`regex`が、信頼区間の計算には`numpy`が必要です。モデルの重みは、論文で引用したモデルカードから入手でき、それぞれの利用条件に従います。

## ライセンス

コードはMITライセンスで公開しています（`LICENSE`を参照）。モデルの重みと診療データは、このライセンスの対象外です。
