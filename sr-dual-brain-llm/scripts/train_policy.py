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

import argparse
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.policy import RightBrainPolicy
try:
    from core.policy_ppo import PPOAgent
    PPO_AVAILABLE = True
except Exception:
    PPO_AVAILABLE = False

DATA = [
    {"q":"今朝のニュースを簡単に教えて","gold":"news summary","difficulty":"easy"},
    {"q":"このデータセットの統計分析結果を詳しく説明してください。","gold":"analysis","difficulty":"hard"},
    {"q":"簡単な要約をして","gold":"short summary","difficulty":"medium"},
]

def grade(ans, gold): return 1.0 if gold in ans else 0.0

def make_state(sample):
    return {"left_conf": 0.45 if sample["difficulty"]=="hard" else 0.9,
            "draft_len": 60, "novelty": 0.0,
            "q_type": ("hard" if sample["difficulty"]=="hard" else "medium")}

def state_to_vec(policy, s): return policy.extract_features(s)

def simulate_outcome(a, sample):
    if a == 0:
        ans = "left-only answer"; acc = 1.0 if sample["difficulty"]!="hard" else 0.0; cost_tokens = 10
    else:
        ans = "left+right answer with analysis " + sample["gold"]; acc = 1.0; cost_tokens = 100
    return ans, acc, cost_tokens

def train_reinforce(epochs=50):
    policy = RightBrainPolicy()
    for ep in range(epochs):
        traj = []
        for _ in range(100):
            sample = random.choice(DATA)
            s = make_state(sample); vec = policy.extract_features(s)
            a = policy.decide(s)
            ans, acc, cost = simulate_outcome(a, sample)
            reward = 1.0*acc - 1e-4*cost
            traj.append((vec, a, reward))
        if hasattr(policy, "reinforce_update") and policy.policy_net is not None:
            loss = policy.reinforce_update(traj); print(f"epoch {ep} loss {loss:.4f}")
        else:
            print(f"epoch {ep} heuristic-based no training")

def train_ppo(epochs=200):
    if not PPO_AVAILABLE:
        print("PPO module not available."); return
    agent = PPOAgent(obs_dim=6, action_dim=3)
    policy = RightBrainPolicy()  # just to reuse feature extractor
    for ep in range(epochs):
        trajectories = []
        for _ in range(200):
            sample = random.choice(DATA)
            s = make_state(sample); obs = state_to_vec(policy, s)
            action, old_logp, _ = agent.select_action(obs)
            ans, acc, cost = simulate_outcome(action, sample)
            reward = 1.0*acc - 1e-4*cost
            trajectories.append({'obs': obs, 'action': action, 'old_logp': old_logp, 'return': reward, 'adv': reward})
        agent.update(trajectories, epochs=4, batch_size=64)
        if ep % 10 == 0: print(f"ppo epoch {ep}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["reinforce","ppo"], default="reinforce")
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()
    if args.algo == "reinforce": train_reinforce(args.epochs)
    else: train_ppo(args.epochs)

if __name__ == "__main__": main()
