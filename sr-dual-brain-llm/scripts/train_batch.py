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

import argparse, json, random
from typing import List, Dict
import torch, torch.nn as nn, torch.optim as optim
def load_trajectories(path: str) -> List[Dict]:
    traj = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ev = json.loads(line)
                if ev.get("type") == "POLICY_DECISION":
                    traj.append(ev)
            except Exception: pass
    return traj
class PolicyNet(nn.Module):
    def __init__(self, in_dim=6, hid=64, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,hid), nn.Tanh(), nn.Linear(hid,hid), nn.Tanh(), nn.Linear(hid,out_dim))
    def forward(self, x):
        if x.dim()==1: x = x.unsqueeze(0)
        return self.net(x)
def to_tensor_state(s: dict):
    import torch
    return torch.tensor([
        float(s.get("left_conf",0.8)),
        float(s.get("draft_len",64))/256.0,
        float(s.get("novelty",0.0)),
        float(s.get("risk",0.0)),
        float(s.get("temp",0.7)),
        1.0 if s.get("q_type","medium")=="hard" else 0.0,
    ], dtype=torch.float32)
def reinforce(traj: List[Dict], epochs=10, lr=1e-3):
    pol = PolicyNet(); opt = optim.Adam(pol.parameters(), lr=lr)
    for ep in range(epochs):
        total_loss = 0.0; n=0
        for ev in traj:
            s = to_tensor_state(ev.get("payload",{}).get("state",{}))
            a = ev.get("payload",{}).get("action",0)
            r = float(ev.get("payload",{}).get("reward",0.0))
            import torch
            logits = pol(s); logp = torch.log_softmax(logits,dim=-1)[0,a]
            loss = -(logp * r)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item()); n+=1
        print(f"[REINFORCE] epoch {ep+1}/{epochs} loss={total_loss/max(1,n):.4f}")
    import torch; torch.save(pol.state_dict(), "policy_reinforce.pt")
def ppo(traj: List[Dict], epochs=10, lr=3e-4, clip_ratio=0.2, batch=64):
    pol = PolicyNet(); opt = optim.Adam(pol.parameters(), lr=lr)
    dataset = []
    import torch
    for ev in traj:
        s = to_tensor_state(ev.get("payload",{}).get("state",{}))
        a = int(ev.get("payload",{}).get("action",0))
        r = float(ev.get("payload",{}).get("reward",0.0))
        logp_old = float(ev.get("payload",{}).get("logp",0.0))
        dataset.append((s,a,r,logp_old))
    for ep in range(epochs):
        random.shuffle(dataset); total_loss=0.0; n=0
        for i in range(0,len(dataset),batch):
            batch_items = dataset[i:i+batch]
            if not batch_items: continue
            states = torch.stack([x[0] for x in batch_items])
            actions = torch.tensor([x[1] for x in batch_items], dtype=torch.long)
            rewards = torch.tensor([x[2] for x in batch_items], dtype=torch.float32)
            logp_old = torch.tensor([x[3] for x in batch_items], dtype=torch.float32)
            logits = pol(states)
            logp = torch.log_softmax(logits,dim=-1).gather(1, actions.view(-1,1)).squeeze(1)
            ratio = torch.exp(logp - logp_old)
            adv = rewards - rewards.mean()
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
            loss = -(torch.min(unclipped, clipped)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item()); n+=1
        print(f"[PPO] epoch {ep+1}/{epochs} loss={total_loss/max(1,n):.4f}")
    torch.save(pol.state_dict(), "policy_ppo.pt")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--algo", choices=["reinforce","ppo"], default="ppo")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    traj = load_trajectories(args.log)
    if not traj: print("No POLICY_DECISION events found in trace."); return
    if args.algo=="reinforce": reinforce(traj, epochs=args.epochs)
    else: ppo(traj, epochs=args.epochs)
if __name__ == "__main__": main()
