# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

AuroLA は、マルチモーダル大規模言語モデル (MLLM) を統一バックボーンとして再利用し、音声とテキストの検索（Audio-Text Retrieval）を行う対照学習フレームワーク。Qwen2.5-Omni (3B/7B) をベースモデルとして使用する。

主な機能:
- **AuroLA**: 音声・テキスト特徴抽出と高速検索のための MLLM ベース埋め込みモデル
- **AuroLA-rerank**: 精密な再ランキングのための MLLM ベース生成モデル

## 環境セットアップ

```bash
pip install -r requirements.txt
```

主要依存: PyTorch 2.5.1, transformers 4.57.1, DeepSpeed 0.17.2, PEFT 0.17.1, accelerate 1.8.1, qwen-omni-utils 0.0.8

## コマンド

### 学習（LoRA ファインチューニング）

```bash
# 検索モデルの事前学習（マルチデータセット対応）
bash scripts/train_retrieval.sh

# 検索モデルのファインチューニング（単一データセット）
bash scripts/finetune_retrieval.sh

# リランキングモデルの学習
bash scripts/train_rerank.sh

# テキスト事前学習
bash scripts/text_pretrain.sh
```

学習は `torchrun` による分散学習で実行される。DeepSpeed ZeRO Stage 2 (`ds_configs/zero2.json`) を使用。

### 評価

```bash
# LoRA チェックポイントの検索テスト
bash scripts/test_retrieval.sh

# HuggingFace モデルの検索テスト
bash scripts/test_retrieval_hfmodel.sh

# リランキングテスト
bash scripts/test_rerank.sh

# リランキングスコアの集計
bash scripts/get_rerank_score.sh
```

各スクリプトは環境変数でパスを設定可能（`MODEL_NAME_OR_PATH`, `AUDIOCAPS_AUDIO_DIR` 等）。

### LoRA マージ

```bash
bash scripts/merge_lora.sh
```

## アーキテクチャ

### モデルレジストリパターン

モデル・コレーター・ローダーはレジストリパターンで管理される:
- `supported_models.py`: `register_model()` でモデルID・ファミリーID・HFパスを登録。`MODULE_KEYWORDS` でモジュール名（vision_encoder, audio_encoder, llm 等）を定義し、学習時のフリーズ対象を制御
- `loaders/__init__.py`: `@register_loader` デコレータで `LOADERS` に登録
- `collators/__init__.py`: `@register_collator` デコレータで `COLLATORS` に登録

新しいモデルを追加するには、これら3箇所すべてに対応エントリを追加する必要がある。

### モデル層 (`model/`)

- `qwen_omni.py`: `Qwen2_50OmniRetForConditionalGeneration` — 推論用モデル。`<emb>` トークン直前の hidden state を埋め込み特徴として抽出。InfoNCE loss で対照学習
- `qwen_omni_finetune.py`: `Qwen2_50OmniRetFinetuneForConditionalGeneration` — 学習用モデル。ミニバッチ分割戦略で GPU メモリ効率化。Hybrid NCE loss（HN-NCE）を実装し、false negative/hard negative に対応。`positive_mask` によるクラスタベースの正例ペア拡張をサポート

### 埋め込み抽出メカニズム

特殊トークン `<emb>` をアシスタント応答に挿入し、その直前位置の hidden state を埋め込みベクトルとして使用。音声側とテキスト側で同じプロンプトテンプレート（`construct_messages()` in `data/dataset_audioverse.py`）を使い、共有空間にマッピング。

### データ層 (`data/`)

- `dataset_audioverse.py`: 学習用データセット。音声ファイルとメタデータJSON からペアを構成。タグクラスタリング/正例マスク機能あり
- `dataset_eval_retrieval.py` / `dataset_eval_rerank.py`: 評価用データセット
- メタデータ形式: `{"video_id": "xxx", "caption": ["..."]}` （`datasets/annotations/` 参照）

### コレーター (`collators/`)

音声・テキストペアをバッチ化。音声の安全性チェック（長さ制限30秒、短い音声のパディング）を実施。`qwen_omni_utils.process_mm_info` でマルチモーダル情報を処理。

### 学習フロー (`train/`)

`train_audio.py` が主要エントリポイント。`HfArgumentParser` で4つの引数グループ（`ModelArguments`, `DataArguments`, `TrainingArguments`, `LoraArguments`）をパース。複数データセットは `--audio_folder` / `--metadata_path` の繰り返しで指定し、`ConcatDataset` で結合。

### 引数体系 (`utils/arguments.py`)

- `ModelArguments`: model_id でレジストリからモデル解決
- `DataArguments`: データパス、ハードネガティブ数、タグクラスタリング設定等
- `TrainingArguments`: contrastive_alpha/lambda/beta（HN-NCE パラメータ）
- `LoraArguments`: LoRA/QLoRA/DoRA の設定、Vision/Audio エンコーダ個別の LoRA rank

## 対応データセット

評価: AudioCaps, Clotho (ClothoV2), Auto-ACD
