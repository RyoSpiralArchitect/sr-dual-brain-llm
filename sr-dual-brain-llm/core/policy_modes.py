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

from dataclasses import dataclass
@dataclass
class DialConfig:
    name: str; consult_bias: float; conf_threshold: float; max_consults: int; temp_multiplier: float; budget: str
MODES = {
    "conservative": DialConfig("conservative", -0.1, 0.5, 0, 0.9, "small"),
    "evaluative":   DialConfig("evaluative",    0.0, 0.7, 1, 1.0, "small"),
    "exploratory":  DialConfig("exploratory",   0.2, 0.85, 2, 1.1, "large"),
}
class ReasoningDial:
    def __init__(self, mode: str = "evaluative"):
        assert mode in MODES, "unknown mode"; self.cfg = MODES[mode]
    def adjust_decision(self, state: dict, action: int) -> int:
        conf = float(state.get("left_conf", 0.8))
        if conf < self.cfg.conf_threshold: action = max(action, 1)
        if self.cfg.consult_bias > 0 and action == 0: action = 1
        if action == 2 and self.cfg.max_consults == 0: action = 1
        return action
    def pick_budget(self) -> str: return self.cfg.budget
    def scale_temperature(self, temp: float) -> float:
        return max(0.1, min(1.2, temp * self.cfg.temp_multiplier))
