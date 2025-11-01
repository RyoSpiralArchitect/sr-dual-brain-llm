# ============================================================================
#  SpiralReality Proprietary
#  Copyright (c) 2025 SpiralReality. All Rights Reserved.
#
#  NOTICE: This file contains confidential and proprietary information of
#  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
#  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
#  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
#
#  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
#  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
#  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
#  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ============================================================================

NEGATIVE_WORDS = ["危険","失敗","不正","リーク","流出","脅威","恐怖","不安","禁止","違反","鍵","パスワード","SECRET","秘密"]
POSITIVE_WORDS = ["成功","安心","達成","良い","うれしい","ありがとう","安全","改善"]
RISK_WORDS     = ["鍵","パスワード","APIキー","認証","個人情報","クレジット","口座","機密","リーク","注入","prompt injection"]

class Amygdala:
    def analyze(self, text: str):
        t = text or ""
        neg = sum(1 for w in NEGATIVE_WORDS if w in t)
        pos = sum(1 for w in POSITIVE_WORDS if w in t)
        risk_hits = sum(1 for w in RISK_WORDS if w in t)
        total = max(1, neg + pos)
        valence = (pos - neg) / total
        arousal = min(1.0, (neg + pos) / 5.0)
        risk = min(1.0, risk_hits / 3.0)
        return {"valence": float(valence), "arousal": float(arousal), "risk": float(risk)}
    def should_escalate(self, metrics):
        return metrics.get("risk",0.0) > 0.66
