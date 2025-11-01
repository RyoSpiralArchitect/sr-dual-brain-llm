# spiral-dual-llm-prototype

Dual-LLM（Left/Right Brain）プロトタイプ — 人工 Corpus Callosum（Callosum）＋共有メモリ＋右脳呼び出しポリシー（ヒューリスティック／RL／PPO）を備えた最小実装バンドル。

## ハイライト
- `core/`：コア実装（Callosum、SharedMemory、Left/RightModel、Policies、Orchestrator、Auditor）
- `core/callosum.py`：in-memory asyncio Callosum（デフォルト）
- `core/callosum_kafka.py`：Kafka ベース Callosum 骨格
- `core/callosum_mqtt.py`：MQTT（Mosquitto）ベース Callosum 骨格
- `core/policy_ppo.py`：PPO（PyTorch）実装（離散 3 アクション）
- `scripts/right_worker_broker.py`：Kafka/MQTT 右脳ワーカー（独立プロセス）
- `scripts/run_server.py`：左脳側サーバ（memory backend では右脳ワーカー内蔵）

## セットアップ（最短）
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.run_server  # in-memory backend
```

## バックエンド切替
環境変数 `CALLOSUM_BACKEND` を使用：`memory`（既定）/ `kafka` / `mqtt`。

```bash
# Kafka
docker-compose up -d   # Kafka & Mosquitto を起動
CALLOSUM_BACKEND=kafka python -m scripts.run_server     # Left側
# 別ターミナルで右脳ワーカーを起動
CALLOSUM_BACKEND=kafka python -m scripts.right_worker_broker
```

```bash
# MQTT (Mosquitto)
docker-compose up -d
CALLOSUM_BACKEND=mqtt python -m scripts.run_server
CALLOSUM_BACKEND=mqtt python -m scripts.right_worker_broker
```

## PPO 学習（サンプル）
```bash
python -m scripts.train_policy --algo ppo --epochs 200
```

## 例
```
Q> このデータセットの統計分析結果を詳しく説明してください。
A> Draft ... + (Reference from RightBrain: Deep analysis ...)
```
