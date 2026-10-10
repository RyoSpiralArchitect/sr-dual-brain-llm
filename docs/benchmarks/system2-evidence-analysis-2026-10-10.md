# System2 と記憶系ベンチの分析 — 2026-10-10

既存の比較では、System2 の安定した回答品質の優位はまだ確認できない。次は多制約の難問で「実際の欠陥を修正した割合」と「正しかった回答を改悪した割合」を独立に測る。その前に、全応答時間を過少計測する箇所を修正する必要がある。

これは既存結果の分析と、外部 LLM を使わない運用診断である。新しいライブモデル比較や人間による採点は実施していない。回答品質、内部診断、計測の健全性をそれぞれ別の根拠として扱う。以下の実装の記述は分析時点の `bbb1459f` に基づく。その後に行った計測修正と固定初稿の比較基盤は [実行手順](frozen-draft-comparison.md) を参照。

## 今回確認できた資料

| 資料 | この checkout で確認できたもの | 分析できる範囲 |
| --- | --- | --- |
| [2026-09-27 A/B](gpt-6-luna-system2-ab-2026-09-27.md) | 文書と圧縮 JSON、off/on 各15ケース | 集計値の再計算、回答の短い preview、計測の不整合 |
| [2026-10-02 paired retest](gpt-6-luna-system2-paired-2026-10-02.md) | 集計文書とファイルのハッシュ記録 | 公開集計の比較。完全回答・trace の再採点はできない |
| [Sol-6.1 の blind review](gpt-6-luna-system2-sol-judge-2026-10-02.md) | 30組の判定表と集計文書 | ケース別の選好と実行順序の関係。採点理由の全文は未入手 |
| 2026-10-10 の heuristic 実行 | creativity 12問、incubation 5×4ターンの新規ローカル結果 | 内部マーカー、キャッシュ、回収経路の動作。モデル品質は採点しない |

09/27 JSON の展開後 SHA-256 は `35111fe5d2334cae729f6e33b0ef6fe3dc512a2aa3a13eda9e5a7cee7cc6fb57`。既存文書の記録と一致した。10/02 の完全回答・blind packets・採点原本は現在の checkout にないため、その数値は文書の記録に基づく。Mistral の非公開スコアは推測せず、人間の採点・裁定も完了結果として扱わない。

計算値とローカル結果のハッシュは [数値記録](system2-evidence-analysis-2026-10-10.metrics.json) に保存した。元のライブ結果と新しいローカル診断を混ぜて集計していない。

## 比較の意味を確認する

`system2_mode=off` でも右脳への通常の相談は可能で、相談の型は `ASK_DETAIL`。`on` は `ASK_CRITIC` に切り替えるほか、初稿への論理優先指示、相談の強制、温度の上限制御、修正ラウンドの予算をまとめて変える。したがって既存の比較は **dual-brain 内で System2 方針を追加した効果**を扱う。左脳単独との比較や、異なるモデル同士の協調効果は測っていない。10/02 の左右のモデルはともに Luna だった。

実装上の根拠は `core/dual_brain.py` の `_run_left_draft_stage`、`_run_routing_stage`、`_build_consult_request`。また `benchmark_system2_ab.py::_run_mode` はモードごとに一つの session を作り、問題間ではメモリをリセットしない。後半の問題には先行 QA や pitfall memory が影響しうる。30組は15種類の問題の反復であり、30種類の独立問題ではない。

## 品質の結果：大部分が同点、差が出たケースも安定しない

10/02 は15問×2モード×2ブロックの60ターンを完了した。30件の `on` は全件 critic valid。ただし valid は応答形式・状態の健全性を示し、critic の指摘が事実として正しいことを保証しない。

| 評価 | 結果 | 読み取れること |
| --- | --- | --- |
| 固定参照の独立採点 | 対象14組中、両側を採点できた12組がすべて correct/correct | この採点可能な範囲では改善・悪化の差なし |
| 固定参照の判定不能 | 2組：AB の `table_inference_001`、BA の `bayes_001` | 認識器が表現を扱えなかった。誤答として数えない |
| Sol の単一モデル判定 | 30組中 on 選好1、off 選好2、同点27 | 90%が同点。二人の独立評者による評価は未完了 |
| Sol の完全性スコア | on/off とも合計57 | 全体の要求充足に安定した差がない |
| 内部 critic issue | AB 6→2、BA 7→5 | 診断上の減少。独立した回答品質の改善量ではない |

固定参照で両側を採点できた12組は、全30組の40%をカバーする。自由回答16組は対象外で、残り2組は判定不能。Sol が全回答に correctness 2/2、unsupported claims 0/2 を付けたことも、評者の判断として扱う。正答率や説明の質を全体で独立検証した結果には置き換えない。

| 差が出たケース | AB（off→on） | BA（on→off） | 解釈 |
| --- | --- | --- | --- |
| `safety_policy_001` | on 選好、完全性差 +1 | off 選好、完全性差 −1 | 要求した4ルールの充足が揺れる。効果の方向は反転 |
| `causal_reasoning_001` | 同点 | off 選好、数値スコア差0 | 粗い3段階スコアに収まらない切り分け案の有用性差 |

非同点の3組はすべて **後に実行したモード**が選好された。これは実行順序を精査する理由になるが、3組だけで順序が原因だと断定できない。今回の同点の多さは、基本問題が容易だった、採点が粗かった、実際に差がなかった、という複数の説明と両立する。「System2 は無効」という同等性の証明でもない。

09/27 の内部 issue 5→4 には、正しい LaTeX の `1/11` への誤指摘と、provider の出力上限失敗が含まれる。`constraints_001` の1→0 は修正成功を調べる候補だが、その診断だけで独立した改善を認定しない。off のケースには issue 測定がなく、summary の合計0を「問題ゼロ」と解釈してはいけない。

## 計測の欠陥：記録された latency は全応答時間ではない

09/27 のケースから集計値は再現できた。ただし phase 合計が `latency_ms` を超えるケースは **28/30**（off 15/15、on 13/15）。単独の phase が `latency_ms` を超えるケースも off 1/15、on 9/15 ある。たとえば on の `arith_chain_001` は初稿生成1,072.84 msに対し、記録された latency は943.34 msだった。

コード上では、controller が初稿生成と routing を済ませた後に `start = time.perf_counter()` を開始し、相談・統合の区間を記録する。後続の observer 等もその区間の外にある。runner の `_run_case` は `controller.process` 全体の `elapsed_ms` を測っているが、telemetry の `latency_ms` があればそちらを採用する。metrics はこの区間を `total_ms` として表示している。

この動作は現 HEAD `bbb1459fc8bc73dcd0740410924c2cfdb7d1b1f5` と、10/02 の実行元 `60d45dcc1b562eeeae4da4b698c04f71a1e09e88` の両方で確認した。09/27 JSON には source revision がなく、その実行コードの厳密な同定はできない。コードの場所は `core/dual_brain.py::process/_run_feedback_consolidation_stage`、`scripts/engine_stdio.py::_extract_metrics`、`scripts/benchmark_system2.py::_run_case`。

| 10/02 の記録値 | off | on | on−off |
| --- | ---: | ---: | ---: |
| AB | 9,035 ms | 6,634 ms | −2,401 ms（−26.6%） |
| BA | 7,457 ms | 8,709 ms | +1,252 ms（+16.8%） |

ブロックごとに方向が反転する。さらに、上記は全応答時間ではない。平均すると後の条件は7,045.5 ms、先の条件は8,872 msだが、この差も実行順序や provider の変動を含む記述値にとどまる。phase の単純合計には未計測区間や重なりがあり、正しい総時間の復元に使わない。今後は runner の外側の wall time を保存し、現在の区間も実際の範囲がわかる名前で別に残す。品質の採点結果はこの時間計測の問題とは別である。

## 記憶・無意識系：回収は動くが、テーマ再訪と期待ラベルに課題

今回、provider/model/key 設定と PG DSN を子プロセスだけで除外し、proxy と TLS 設定を保った heuristic 実行を行った。両スクリプトは exit code 0。リポジトリの `.venv/bin/python` を使い、標準 fixture の全件をファイル順に実行した。plain 回答、leading/system2 auto、executive/observer off、low-signal filter on、`--expect-no-leaks --history ''` を使用した。実際の config と runtime バージョンも数値記録に残した。外部モデルの品質比較ではない。

| 観測 | Creativity | Incubation |
| --- | ---: | ---: |
| 完了 | 12/12問 | 5/5シーケンス、20/20ターン |
| エラー | 0 | 0 |
| 固定内部マーカーの検出 | 0 | 0 |
| emergence | 0/12問 | 5/5シーケンス |
| 想定 target の emergence | 対象外 | 1/5 |
| 完全反復 echo で emergence | 対象外 | 5/5 |
| 要求を変えたテーマ再訪 return で emergence | 対象外 | 0/5 |

| シーケンス | 想定 target（どれか1つ） | 回収されたラベル |
| --- | --- | --- |
| mirror | shadow / syzygy | syzygy |
| forest | great_mother / hero | syzygy |
| mask | persona | self |
| trickster | hero / trickster | syzygy |
| child | child / self | shadow |

全件の emergence は4ターン目の完全反復で発生し、echo の類似度は0.9727〜0.9796。3ターン目の return は同テーマに別の要求を加えた質問であり、同じ問いの単純な言い換えではない。内容だけでなく、履歴の位置と蓄積量も異なるので、0/5対5/5から言い換えへの弱さや完全反復が原因であることを切り分けられない。次は同じ cache 状態から条件を分岐させるか、return/echo の位置を均衡化する。

lineage の同 archetype・origin 一致は5/5だが、事後の文字列・ラベル照合による対応付けであり、因果的な seed ID の追跡ではない。小さなこの fixture では、回収成功と想定ラベルの一致が分かれた。target は fixture が定めた期待で、回答の有用性を独立に採点したものではない。

現実装の emergent idea は、保存した archetype ラベル・強度・類似度・元質問の接頭辞を表す。新しい提案本文の独立評価ではない。creativity score も内部信号の加重和である。漏れ0は14個の固定マーカーがなかったという結果で、あらゆる表現の内部情報混入を検査したわけではない。これらの意味は [既存の説明](../unconscious_benchmark_results.md) と実装に基づく。

## 次の実験の優先順位

1. **先に時間計測を修正する。** full wall time と相談区間を分け、初稿が遅いケースでも全体時間に含まれる検証を加える。今回の分析ではコードの変更は行っていない。
2. **同じ初稿で役割の効果を比べる。** 初稿を固定し、修正なし／左脳の自己検証＋修正／右脳の批評＋左脳の修正を比較する。自己検証と右脳批評は呼び出し数・token上限を揃え、実 token 使用量、wall time、失敗・打ち切りも記録する。実消費が違う場合もそのまま報告する。
3. **4種類の多制約問題に絞る。** `concurrency_review_001`、`incident_causal_triage_002`、`security_logging_policy_002`、`distributed_consistency_001` で、生成前に必須条件を採点表として固定する。通常初稿の修正と、既知の欠陥を入れた初稿の修正は別の実験として扱う。後者の成功率を自然発生の改善率へ読み替えない。
4. **記憶の有用性と汚染を比較する。** 同じ最終問題を、fresh session／正しい先行情報／古い誤情報／無関係な先行情報で比べる。完全反復、同じ問いの言い換え、モチーフ名なしの再参照を、同じ履歴位置・先行 cache 状態で別条件にする。回収だけでなく、回答の制約充足と誤った前提の持ち込みを採点する。

新しい比較では条件の実行順を事前にランダム化・均衡化し、反復数と問題リストを先に固定する。単問の比較では session を分け、履歴を持つ実験とは区別する。匿名の独立採点と不一致の裁定は [既存プロトコル](system2-paired-evaluation-protocol.md) を使う。主要指標は、実在する欠陥を直した割合、正しい初稿を改悪した割合、必須条件の充足、裏付けのない断言、wall time と token 使用量。修正の前後、critic の根拠、検証結果はローカル trace に残す。

### 問題セットの注意点

既存5セットは合計291エントリだが、IDは204種類、質問本文は196種類。同じIDで異なる本文のケースはなかった。base の15問は XL と XL2 にも含まれる。タグも重複するため、連結件数を独立問題数として扱わない。

hard の `simpson_paradox_001` は A/B の mobile が2%/4%、desktop が4%/6%、全体が3.538%/5.667%で、すべて B が上回る。Simpson の逆転は起きていない。「逆転がある」という前提を退けられるかを測る control として扱うか、別の変更でデータを修正する。今回は fixture を変更していない。

## 再計算とローカル成果物

09/27 の数値検証は Python 標準ライブラリだけで再現できる。リポジトリルートで実行する。

```python
import gzip, hashlib, json, math, statistics
from pathlib import Path

raw = gzip.decompress(Path("docs/benchmarks/gpt-6-luna-system2-ab-2026-09-27.json.gz").read_bytes())
assert hashlib.sha256(raw).hexdigest() == "35111fe5d2334cae729f6e33b0ef6fe3dc512a2aa3a13eda9e5a7cee7cc6fb57"
report = json.loads(raw)
for mode in ("off", "on"):
    cases = report["modes"][mode]["cases"]
    mean = statistics.mean(c["latency_ms"] for c in cases)
    assert math.isclose(mean, report["summary_by_mode"][mode]["avg_latency_ms_all_cases"])
    excess = sum(sum(c["phase_latency_ms"].values()) > c["latency_ms"] for c in cases)
    print(mode, len(cases), round(mean, 3), excess)
```

出力は `off 15 4592.871 15`、`on 15 2988.676 13`。これは記録された区間の再計算で、正しい全応答時間への補正ではない。

新規の raw JSON とログは Git ignored の `target/benchmarks/analysis-2026-10-10/` にある。分析用の集計値・ハッシュだけを数値記録に残し、回答本文や trace は追加していない。分析の段階ではこの文書と数値記録のみを追加し、既存のアプリケーション、テスト、fixture、過去レポートは変更していない。その後の実装作業は上記の実行手順に記載している。
