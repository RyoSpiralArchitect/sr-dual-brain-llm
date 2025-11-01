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

import random
from typing import Dict, Any
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

class RightBrainPolicy:
    """Heuristic policy with optional REINFORCE hook (if torch available)."""
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        if TORCH_AVAILABLE:
            self.policy_net = nn.Sequential(
                nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 3), nn.Softmax(dim=-1)
            )
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        else:
            self.policy_net = None

    def extract_features(self, state: Dict[str, Any]):
        left_conf = float(state.get("left_conf", 0.5))
        draft_len = float(state.get("draft_len", 10))/100.0
        novelty = float(state.get("novelty", 0.0))
        qtype = state.get("q_type", "other")
        onehot = [0.0,0.0,0.0]
        if qtype=="easy": onehot[0]=1.0
        elif qtype=="medium": onehot[1]=1.0
        elif qtype=="hard": onehot[2]=1.0
        return [left_conf, draft_len, novelty]+onehot

    def decide(self, state: Dict[str, Any]) -> int:
        vec = self.extract_features(state)
        if self.policy_net is None:
            if state.get("left_conf",0.5) < self.threshold or state.get("q_type","other")=="hard":
                return 1
            return 0
        else:
            import torch
            with torch.no_grad():
                x = torch.tensor([vec], dtype=torch.float32)
                probs = self.policy_net(x).cpu().numpy()[0]
                return int(random.choices(range(3), probs)[0])

    def reinforce_update(self, trajectories):
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available")
        import torch
        losses = []
        for svec, action, reward in trajectories:
            x = torch.tensor([svec], dtype=torch.float32)
            probs = self.policy_net(x)
            logp = torch.log(probs[0, action] + 1e-8)
            losses.append(-logp * reward)
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
