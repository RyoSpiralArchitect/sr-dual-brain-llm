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

class Hypothalamus:
    def __init__(self, ema_beta: float = 0.9, base_slot_ms: int = 250):
        self.ema_beta = ema_beta
        self.base_slot_ms = base_slot_ms
        self._reward_avg = 0.5
        self._latency_avg_ms = 1000.0
    def update_feedback(self, reward: float, latency_ms: float):
        self._reward_avg = self.ema_beta * self._reward_avg + (1 - self.ema_beta) * max(0.0, min(1.0, reward))
        self._latency_avg_ms = self.ema_beta * self._latency_avg_ms + (1 - self.ema_beta) * max(0.0, latency_ms)
    @property
    def satiety(self) -> float: return self._reward_avg
    @property
    def hunger(self) -> float: return 1.0 - self.satiety
    def recommend_temperature(self, left_conf: float) -> float:
        temp = 0.3 + 0.6 * max(0.0, 1.0 - left_conf) + 0.3 * self.hunger
        return float(max(0.3, min(0.95, temp)))
    def recommend_slot_ms(self, risk: float) -> int:
        widen = int(100 * max(0.0, min(1.0, risk)))
        return int(self.base_slot_ms + widen)
    def bias_for_consult(self, novelty: float) -> float:
        return max(-0.2, min(0.2, 0.1*self.hunger + 0.1*max(0.0, min(1.0, novelty))))
