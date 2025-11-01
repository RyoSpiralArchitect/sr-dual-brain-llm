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

import numpy as np
import torch, torch.nn as nn, torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=6, action_dim=3, hidden=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.policy = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim), nn.Softmax(dim=-1))
        self.value_head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.fc(x)
        pi = self.policy(h)
        v = self.value_head(h)
        return pi, v.squeeze(-1)

class PPOAgent:
    def __init__(self, obs_dim=6, action_dim=3, lr=3e-4, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
        self.net = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.clip_eps = clip_eps; self.vf_coef = vf_coef; self.ent_coef = ent_coef

    def select_action(self, obs):
        import numpy as np
        obs_t = torch.tensor([obs], dtype=torch.float32)
        with torch.no_grad():
            pi, v = self.net(obs_t)
        probs = pi.cpu().numpy()[0]
        action = int(np.random.choice(len(probs), p=probs))
        return action, float(np.log(max(probs[action],1e-8))), float(v.cpu().numpy()[0])

    def update(self, trajectories, epochs=4, batch_size=64):
        import numpy as np, torch
        obs = torch.tensor(np.vstack([t['obs'] for t in trajectories]), dtype=torch.float32)
        actions = torch.tensor([t['action'] for t in trajectories], dtype=torch.long)
        old_logp = torch.tensor([t['old_logp'] for t in trajectories], dtype=torch.float32)
        returns = torch.tensor([t['return'] for t in trajectories], dtype=torch.float32)
        advs = torch.tensor([t['adv'] for t in trajectories], dtype=torch.float32)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        N = len(trajectories)
        for _ in range(epochs):
            perm = np.random.permutation(N)
            for i in range(0, N, batch_size):
                idx = perm[i:i+batch_size]
                b_obs, b_actions = obs[idx], actions[idx]
                b_old_logp, b_returns, b_advs = old_logp[idx], returns[idx], advs[idx]
                pi, values = self.net(b_obs)
                dist = torch.distributions.Categorical(pi)
                new_logp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()
                ratio = (new_logp - b_old_logp).exp()
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (b_returns - values).pow(2).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
