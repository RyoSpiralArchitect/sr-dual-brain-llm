# Spiral Dual-LLM Integrated Bundle

Neuro Add-on（Hypothalamus / Amygdala / Temporal Hippocampal Indexing）と
RTX Addon（Replay/Logging + Dashboard, Trainer Loop, Reasoning Dial）の統合バンドル。

## すぐ試す
1) サンプル実行（trace生成）
    python -m scripts.run_server_integrated
2) 可視化
    pip install fastapi uvicorn
    python -m scripts.replay_timeline --log traces/session.jsonl --host 127.0.0.1 --port 8765
3) 学習（PPO）
    pip install torch
    python -m scripts.train_batch --log traces/session.jsonl --algo ppo --epochs 10

各ファイルには Proprietary ヘッダを付与済み。
最終更新: 2025-11-01T07:19:29.131250Z
