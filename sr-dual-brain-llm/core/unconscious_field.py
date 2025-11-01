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

"""
Unconscious Field (Archetopos Integration)
==========================================

A pluggable subsystem that maps narratives to archetypal intensities and
provides a compact summary suitable for enriching the dual-brain controller
with an "unconscious" perspective.

Highlights (v0.2-inspired):
- Deterministic hash-based embedding with optional NumPy/Matplotlib upgrades.
- Tunable softmax temperature, embedding dimension, and curvature probing.
- Prototype catalogue with JSON persistence helpers.
- Polar projection for trajectory inspection (if Matplotlib is available).
- CLI-compatible entry points for demo, mapping, plotting, and tuning.

Author: SpiralReality / Archetopos Team
Version: 0.2-adapted
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import random
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Optional imports (graceful)
    import numpy as np  # type: ignore

    HAVE_NUMPY = True
except Exception:  # pragma: no cover
    HAVE_NUMPY = False

try:  # Optional imports (graceful)
    import matplotlib.pyplot as plt  # type: ignore

    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Prototype:
    id: str
    label: str
    keywords: List[str]
    notes: str = ""


@dataclass
class Geometry:
    r: float
    theta: float
    curvature_proxy: float


@dataclass
class ArchetypeScore:
    id: str
    label: str
    similarity: float
    intensity: float


@dataclass
class EventMapping:
    event_id: str
    text: str
    embedding_dim: int
    geometry: Geometry
    archetype_map: List[ArchetypeScore]
    top_k: List[str]


@dataclass
class LatentSeed:
    """Cached trace for unresolved material incubating in the unconscious."""

    question: str
    draft: str
    archetype_id: str
    archetype_label: str
    intensity: float
    novelty: float
    vector: List[float]
    created_at: float
    exposures: int = 0

    def short_origin(self) -> str:
        base = self.question.strip() or self.draft.strip()
        return base[:120]


@dataclass
class EmergentIdea:
    """Representation of a cached seed that resurfaced with a new insight."""

    archetype: str
    label: str
    intensity: float
    incubation_rounds: int
    trigger_similarity: float
    origin: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "archetype": self.archetype,
            "label": self.label,
            "intensity": self.intensity,
            "incubation_rounds": self.incubation_rounds,
            "trigger_similarity": self.trigger_similarity,
            "origin": self.origin,
        }


# ---------------------------------------------------------------------------
# Psychoid archetype projection
# ---------------------------------------------------------------------------


@dataclass
class PsychoidSignal:
    """Latent projection describing psychoid archetype influences."""

    attention_bias: List[Dict[str, object]]
    bias_vector: List[float]
    psychoid_tension: float
    resonance: float
    signifier_chain: List[str]


class PsychoidArchetypeSampler:
    """Project archetypal intensities into an attention-bias style signal."""

    def __init__(
        self,
        prototypes: Dict[str, Prototype],
        *,
        qkv_dim: int = 16,
        max_chain: int = 24,
        smoothing: float = 0.72,
    ) -> None:
        self.prototypes = prototypes
        self.qkv_dim = max(4, qkv_dim)
        self.smoothing = smoothing
        self._chain: Deque[str] = deque(maxlen=max_chain)
        self._latent_resonance = 0.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _keyword_overlap(self, archetype_id: str, tokens: Sequence[str]) -> Tuple[int, int]:
        proto = self.prototypes.get(archetype_id)
        if not proto:
            return 0, 0
        keywords = [kw.lower() for kw in proto.keywords]
        if not keywords:
            return 0, 0
        overlap = sum(1 for kw in keywords if kw in tokens)
        return overlap, len(keywords)

    def sample_signal(
        self,
        mapping: EventMapping,
        *,
        question: str,
        draft: Optional[str] = None,
        qkv_dim: Optional[int] = None,
    ) -> PsychoidSignal:
        qkv = max(4, qkv_dim or self.qkv_dim)
        tokens = self._tokenize(f"{question} {draft or ''}")
        weighted: List[Tuple[ArchetypeScore, float, float]] = []
        for score in mapping.archetype_map:
            overlap, keyword_total = self._keyword_overlap(score.id, tokens)
            resonance = 0.0
            if keyword_total:
                resonance = overlap / keyword_total
            psychoid_weight = score.intensity * (1.0 + 0.6 * resonance)
            weighted.append((score, psychoid_weight, resonance))

        weighted.sort(key=lambda item: item[1], reverse=True)
        total_weight = sum(item[1] for item in weighted) or 1.0
        attention_bias: List[Dict[str, object]] = []
        bias_vector: List[float] = []
        top_entries = weighted[: max(1, min(4, len(weighted)))]
        for score, weight, resonance in top_entries:
            attention_bias.append(
                {
                    "archetype": score.id,
                    "label": score.label,
                    "weight": float(weight / total_weight),
                    "resonance": float(resonance),
                }
            )
        if attention_bias:
            for idx in range(qkv):
                entry = attention_bias[idx % len(attention_bias)]
                bias_vector.append(float(entry["weight"]))
        else:
            bias_vector = [0.0 for _ in range(qkv)]

        psychoid_tension = 0.0
        if weighted:
            peak = weighted[0][1]
            trough = weighted[-1][1]
            psychoid_tension = float(max(0.0, peak - trough))
        max_resonance = max((item[2] for item in weighted), default=0.0)
        self._latent_resonance = (
            self.smoothing * self._latent_resonance + (1.0 - self.smoothing) * max_resonance
        )

        if top_entries:
            top_score, _, _ = top_entries[0]
            overlap_tokens = [
                kw
                for kw in self.prototypes.get(top_score.id, Prototype("", "", [])).keywords
                if kw.lower() in tokens
            ]
            excerpt = overlap_tokens[0] if overlap_tokens else top_score.label
            self._chain.append(f"{top_score.id}:{excerpt}")

        return PsychoidSignal(
            attention_bias=attention_bias,
            bias_vector=bias_vector,
            psychoid_tension=psychoid_tension,
            resonance=float(self._latent_resonance),
            signifier_chain=list(self._chain),
        )

    def integrate_feedback(self, *, success: bool, reward: Optional[float]) -> None:
        decay = 0.78 if success else 0.62
        self._latent_resonance *= decay
        if reward is not None:
            self._latent_resonance += 0.12 * max(0.0, 0.75 - reward)
        self._latent_resonance = max(0.0, min(1.0, self._latent_resonance))
        if not success:
            self._chain.append("rupture")


# ---------------------------------------------------------------------------
# Embedding (pluggable)
# ---------------------------------------------------------------------------


class TextEmbedder:
    """Interface for text embedding."""

    def encode(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover - interface
        raise NotImplementedError


class HashEmbedder(TextEmbedder):
    """Deterministic pseudo-embedding via hashing (dependency-free)."""

    def __init__(self, dim: int = 128, slots: int = 8, seed: int = 0):
        self.dim = dim
        self.slots = slots
        self.seed = seed

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _vec(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        vec = [0.0] * self.dim
        for tok in tokens:
            h = _sha256_hex(tok)
            mix = int(h[:16], 16) ^ (self.seed & 0xFFFFFFFF)
            rng = random.Random(mix)
            for _ in range(self.slots):
                idx = rng.randrange(self.dim)
                val = rng.random() * 2.0 - 1.0
                vec[idx] += val
        return _l2_normalize(vec)

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _sha256_hex(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _l2(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _l2_normalize(v: Sequence[float]) -> List[float]:
    n = _l2(v) or 1.0
    return [x / n for x in v]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    denom = (_l2(a) * _l2(b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / denom


def softmax(xs: Sequence[float], beta: float) -> List[float]:
    m = max(xs)
    exps = [math.exp(beta * (x - m)) for x in xs]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


# ---------------------------------------------------------------------------
# Prototypes I/O and defaults
# ---------------------------------------------------------------------------


DEFAULT_PROTOTYPES: Dict[str, Prototype] = {
    "self": Prototype(
        id="self",
        label="Self",
        keywords=[
            "center",
            "circle",
            "mandala",
            "integration",
            "quaternity",
            "coordinate_origin",
            "equilibrium",
            "taiji",
            "spiritual_center",
        ],
        notes="Integrative core; potential well minimum.",
    ),
    "persona": Prototype(
        id="persona",
        label="Persona",
        keywords=[
            "mask",
            "stage",
            "role",
            "business_card",
            "presentation",
            "festival",
            "society",
            "disguise",
            "spotlight",
            "reputation",
            "manners",
        ],
        notes="Social interface boundary with pronounced curvature.",
    ),
    "shadow": Prototype(
        id="shadow",
        label="Shadow",
        keywords=[
            "shadow",
            "pursuit",
            "hunter",
            "darkness",
            "monster",
            "fear",
            "concealment",
            "conflict",
            "denial",
            "envy",
            "destructive_impulse",
        ],
        notes="Negative curvature; bridge-like edges with tension.",
    ),
    "syzygy": Prototype(
        id="syzygy",
        label="Syzygy (Anima/Animus)",
        keywords=[
            "mirror",
            "water",
            "surface",
            "moon",
            "twin",
            "pair",
            "intersection",
            "fusion",
            "symmetry",
            "bride",
            "groom",
            "moonlight",
        ],
        notes="Double helix torsion with heightened twist.",
    ),
    "sage": Prototype(
        id="sage",
        label="Sage",
        keywords=[
            "sage",
            "mentor",
            "book",
            "library",
            "lighthouse",
            "guidepost",
            "insight",
            "prophecy",
            "mountaintop",
        ],
        notes="Knowledge hub with high betweenness centrality.",
    ),
    "trickster": Prototype(
        id="trickster",
        label="Trickster",
        keywords=[
            "fox",
            "jester",
            "transformation",
            "mischief",
            "misdirection",
            "reversal",
            "boundary_crossing",
            "chaos",
            "disruption",
        ],
        notes="Short-period loops and local phase transitions.",
    ),
    "hero": Prototype(
        id="hero",
        label="Hero",
        keywords=[
            "sword",
            "shield",
            "gate",
            "trial",
            "dragon",
            "journey",
            "quest",
            "return",
            "courage",
            "fire",
            "crossing",
            "victory_cry",
        ],
        notes="Momentum toward overcoming potential barriers.",
    ),
    "great_mother": Prototype(
        id="great_mother",
        label="Great Mother",
        keywords=[
            "mother",
            "nurturing",
            "sea",
            "earth",
            "cave",
            "embrace",
            "womb",
            "milk",
            "abundance",
            "harvest",
            "soil",
            "shell",
        ],
        notes="Enveloping basin; bowl-shaped potential well.",
    ),
    "child": Prototype(
        id="child",
        label="Child",
        keywords=[
            "egg",
            "sprout",
            "rebirth",
            "dawn",
            "hope",
            "innocence",
            "play",
            "birth",
        ],
        notes="Generative endpoint with positive curvature.",
    ),
}


def load_prototypes(path: Optional[str]) -> Dict[str, Prototype]:
    if not path:
        return dict(DEFAULT_PROTOTYPES)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prototype file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        items = raw.items()
    else:
        items = [(r.get("id"), r) for r in raw]
    out: Dict[str, Prototype] = {}
    for pid, rec in items:
        if not pid:
            continue
        out[pid] = Prototype(
            id=pid,
            label=rec.get("label", pid),
            keywords=list(rec.get("keywords", [])),
            notes=rec.get("notes", ""),
        )
    return out


def save_prototypes(path: str, protos: Dict[str, Prototype]) -> None:
    arr = [dataclasses.asdict(p) for p in protos.values()]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Core mapping pipeline
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    dim: int = 128
    beta: float = 10.0
    k_curv: int = 3
    seed: int = 0


class Archetopos:
    def __init__(
        self,
        prototypes: Dict[str, Prototype],
        cfg: PipelineConfig,
        embedder: Optional[TextEmbedder] = None,
    ):
        self.prototypes = prototypes
        self.cfg = cfg
        self.embedder = embedder or HashEmbedder(dim=cfg.dim, seed=cfg.seed)
        self._proto_ids: List[str] = list(prototypes.keys())
        proto_texts = [" ".join(prototypes[pid].keywords) for pid in self._proto_ids]
        self._proto_vecs = self.embedder.encode(proto_texts)

    def map_events(self, texts: Sequence[str]) -> List[EventMapping]:
        vecs = self.embedder.encode(texts)
        sims = self._pairwise_proto_sims(vecs)
        intensities = [softmax(row, beta=self.cfg.beta) for row in sims]
        geom = self._geometry(vecs)
        out: List[EventMapping] = []
        for i, txt in enumerate(texts):
            amap = [
                ArchetypeScore(
                    id=self._proto_ids[j],
                    label=self.prototypes[self._proto_ids[j]].label,
                    similarity=float(sims[i][j]),
                    intensity=float(intensities[i][j]),
                )
                for j in range(len(self._proto_ids))
            ]
            amap.sort(key=lambda x: x.intensity, reverse=True)
            topk = [a.id for a in amap[:3]]
            out.append(
                EventMapping(
                    event_id=f"E{i+1:02d}",
                    text=txt,
                    embedding_dim=self.cfg.dim,
                    geometry=geom[i],
                    archetype_map=amap,
                    top_k=topk,
                )
            )
        return out

    def _pairwise_proto_sims(self, vecs: Sequence[Sequence[float]]) -> List[List[float]]:
        sims: List[List[float]] = []
        for v in vecs:
            row = [cosine(v, p) for p in self._proto_vecs]
            sims.append(row)
        return sims

    def _geometry(self, vecs: Sequence[Sequence[float]]) -> List[Geometry]:
        sim_matrix = _pairwise_sims(vecs)
        curvatures = [_curvature_proxy(sim_matrix, i, k=self.cfg.k_curv) for i in range(len(vecs))]
        xy_points = _project_2d(vecs)
        out: List[Geometry] = []
        for (x, y), kappa in zip(xy_points, curvatures):
            r = math.sqrt(x * x + y * y)
            theta = math.atan2(y, x)
            out.append(Geometry(r=r, theta=theta, curvature_proxy=kappa))
        return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _pairwise_sims(vecs: Sequence[Sequence[float]]) -> List[List[float]]:
    n = len(vecs)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = cosine(vecs[i], vecs[j])
            matrix[i][j] = s
            matrix[j][i] = s
    return matrix


def _curvature_proxy(sim_matrix: List[List[float]], i: int, k: int = 3) -> float:
    sims = [(s, j) for j, s in enumerate(sim_matrix[i]) if j != i]
    sims.sort(key=lambda t: t[0], reverse=True)
    top = [s for s, _ in sims[:k]] or [0.0]
    far = [s for s, _ in sims[-k:]] or [0.0]
    return (sum(top) / len(top)) - (sum(far) / len(far))


def _project_2d(vecs: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    if HAVE_NUMPY:  # PCA via SVD
        X = np.array(vecs, dtype=float)
        Xc = X - X.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        take = min(2, Vt.shape[0])
        coords = Xc @ Vt.T[:, :take]
        if coords.shape[1] < 2:
            coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])), mode="constant")
        norms = np.linalg.norm(coords, axis=1, keepdims=True) + 1e-9
        coords = coords / norms
        return [(float(c[0]), float(c[1])) for c in coords]
    out: List[Tuple[float, float]] = []
    for v in vecs:
        x = sum(v[::2])
        y = sum(v[1::2])
        n = math.sqrt(x * x + y * y) or 1.0
        out.append((x / n, y / n))
    return out


# ---------------------------------------------------------------------------
# JSON/CSV/Plot outputs
# ---------------------------------------------------------------------------


def export_json(path: str, events: List[EventMapping], prototypes: Dict[str, Prototype]) -> None:
    payload = {
        "version": "0.2",
        "generated_at": _utcnow(),
        "events": [
            {
                "event_id": ev.event_id,
                "timestamp": _utcnow(),
                "modality": "text",
                "text": ev.text,
                "embedding_dim": ev.embedding_dim,
                "geometry": dataclasses.asdict(ev.geometry),
                "archetype_map": [dataclasses.asdict(a) for a in ev.archetype_map],
                "top_k": list(ev.top_k),
                "links": [],
            }
            for ev in events
        ],
        "archetype_definitions": [dataclasses.asdict(p) for p in prototypes.values()],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def export_csv(path: str, events: List[EventMapping]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "event_id",
                "top1",
                "top1_intensity",
                "top2",
                "top2_intensity",
                "top3",
                "top3_intensity",
                "r",
                "theta",
                "curvature_proxy",
                "text",
            ]
        )
        for ev in events:
            top = ev.archetype_map[:3]
            writer.writerow(
                [
                    ev.event_id,
                    getattr(top[0], "id", ""),
                    f"{getattr(top[0], 'intensity', 0.0):.4f}",
                    getattr(top[1], "id", ""),
                    f"{getattr(top[1], 'intensity', 0.0):.4f}",
                    getattr(top[2], "id", ""),
                    f"{getattr(top[2], 'intensity', 0.0):.4f}",
                    f"{ev.geometry.r:.4f}",
                    f"{ev.geometry.theta:.4f}",
                    f"{ev.geometry.curvature_proxy:.4f}",
                    ev.text,
                ]
            )


def plot_polar(path: str, events: List[EventMapping], title: str = "SpiralArchetopos — Polar (r, θ)") -> None:
    if not HAVE_MPL:  # pragma: no cover
        raise RuntimeError("matplotlib is not available for plotting")
    thetas = [ev.geometry.theta for ev in events]
    rs = [ev.geometry.r for ev in events]
    labels = [ev.event_id for ev in events]
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")
    ax.plot(thetas, rs, marker="o")
    for th, r, lab in zip(thetas, rs, labels):
        ax.text(th, r, lab)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# Convenience / CLI
# ---------------------------------------------------------------------------


DEMO_EVENTS = [
    "In a dream, someone chases me through a dark forest. My legs grow heavier the more I run.",
    "At a festival I dance with a mask and laugh with the crowd.",
    "In an old library a mentor-like woman shows me the way.",
    "At the bottom of the sea, a gigantic motherly presence holds me in a cave.",
    "When I fix a broken clock, time runs backward and I meet my younger self.",
    "A cunning fox offers to guide me but leads me astray on purpose.",
    "Standing at the center of a meditation circle, I feel united with the world.",
    "I face the shadow of a dragon while holding a sword.",
]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f]
    return [ln for ln in lines if ln]


def cmd_demo(args: argparse.Namespace) -> None:
    protos = load_prototypes(args.prototypes)
    cfg = PipelineConfig(dim=args.dim, beta=args.beta, k_curv=args.k, seed=args.seed)
    arc = Archetopos(protos, cfg)
    events = arc.map_events(DEMO_EVENTS)
    if args.out_json:
        export_json(args.out_json, events, protos)
    if args.out_csv:
        export_csv(args.out_csv, events)
    if args.plot:
        plot_polar(args.plot, events, title=f"SpiralArchetopos — Polar (β={args.beta})")
    _print_summary(events)


def cmd_map(args: argparse.Namespace) -> None:
    if args.texts:
        texts = _read_lines(args.texts)
    else:
        print("[map] No --texts provided; reading stdin (one line per event)...", file=sys.stderr)
        texts = [ln.strip() for ln in sys.stdin if ln.strip()]
    protos = load_prototypes(args.prototypes)
    cfg = PipelineConfig(dim=args.dim, beta=args.beta, k_curv=args.k, seed=args.seed)
    arc = Archetopos(protos, cfg)
    events = arc.map_events(texts)
    if args.out_json:
        export_json(args.out_json, events, protos)
    if args.out_csv:
        export_csv(args.out_csv, events)
    if args.plot:
        plot_polar(args.plot, events, title=f"SpiralArchetopos — Polar (β={args.beta})")
    _print_summary(events)


def cmd_plot(args: argparse.Namespace) -> None:
    if not args.in_json:
        raise ValueError("--in-json is required for plot command")
    with open(args.in_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    events = []
    for ev in payload.get("events", []):
        geom = ev.get("geometry", {})
        g = Geometry(
            r=float(geom.get("r", 0.0)),
            theta=float(geom.get("theta", 0.0)),
            curvature_proxy=float(geom.get("curvature_proxy", 0.0)),
        )
        amap = [ArchetypeScore(**a) for a in ev.get("archetype_map", [])]
        events.append(
            EventMapping(
                event_id=ev.get("event_id", "E00"),
                text=ev.get("text", ""),
                embedding_dim=int(ev.get("embedding_dim", 0)),
                geometry=g,
                archetype_map=amap,
                top_k=list(ev.get("top_k", [])),
            )
        )
    if args.plot:
        plot_polar(args.plot, events, title=args.title or "SpiralArchetopos — Polar")


def cmd_tune(args: argparse.Namespace) -> None:
    protos = load_prototypes(args.prototypes)
    if args.add:
        pid = args.add[0]
        extra = args.add[1:]
        if pid not in protos:
            protos[pid] = Prototype(id=pid, label=pid, keywords=[])
        merged = list(dict.fromkeys(list(protos[pid].keywords) + list(extra)))
        protos[pid].keywords = merged
        print(f"[tune] Added {len(extra)} keywords to '{pid}'.")
    if args.save_to:
        save_prototypes(args.save_to, protos)
        print(f"[tune] Saved prototypes -> {args.save_to}")
    else:
        print(json.dumps([dataclasses.asdict(p) for p in protos.values()], ensure_ascii=False, indent=2))


def _print_summary(events: List[EventMapping]) -> None:
    print("Top-1 archetype per event:")
    for ev in events:
        top1 = ev.archetype_map[0]
        print(f"  {ev.event_id}: {top1.label} ({top1.id}) — intensity={top1.intensity:.3f}")


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SpiralArchetopos v0.2 — map narratives to archetypal fields"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--prototypes", type=str, default=None, help="Path to prototypes.json")
        sp.add_argument("--dim", type=int, default=128, help="Embedding dimension for HashEmbedder")
        sp.add_argument("--beta", type=float, default=10.0, help="Softmax temperature")
        sp.add_argument("--k", type=int, default=3, help="k for curvature proxy")
        sp.add_argument("--seed", type=int, default=0, help="Embedding RNG seed mix-in")
        sp.add_argument("--out-json", type=str, default=None, help="Export mapping JSON path")
        sp.add_argument("--out-csv", type=str, default=None, help="Export mapping CSV path")
        sp.add_argument("--plot", type=str, default=None, help="Save polar plot PNG path")

    demo = sub.add_parser("demo", help="Run on built-in demo events")
    add_common(demo)

    map_parser = sub.add_parser("map", help="Map events from --texts file or stdin")
    add_common(map_parser)
    map_parser.add_argument(
        "--texts",
        type=str,
        default=None,
        help="UTF-8 text file (one event per line); if omitted, read stdin",
    )

    plot_parser = sub.add_parser("plot", help="Plot polar from an existing mapping JSON")
    plot_parser.add_argument("--in-json", type=str, required=True, help="Input JSON path")
    plot_parser.add_argument("--plot", type=str, required=True, help="Output PNG path")
    plot_parser.add_argument("--title", type=str, default=None, help="Optional plot title")

    tune = sub.add_parser("tune", help="Modify prototypes (add keywords; print or save)")
    tune.add_argument("--prototypes", type=str, default=None, help="Path to prototypes.json")
    tune.add_argument("--add", nargs="+", help="Usage: --add <proto_id> <kw1> <kw2> ...")
    tune.add_argument("--save-to", type=str, default=None, help="Write prototypes to this path")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "demo":
        cmd_demo(args)
    elif args.cmd == "map":
        cmd_map(args)
    elif args.cmd == "plot":
        cmd_plot(args)
    elif args.cmd == "tune":
        cmd_tune(args)
    else:  # pragma: no cover
        parser.print_help()
        return 2
    return 0


# ---------------------------------------------------------------------------
# Integration helper for the dual-brain controller
# ---------------------------------------------------------------------------


class UnconsciousField:
    """Wrapper that provides archetypal summaries for controller integration."""

    def __init__(
        self,
        *,
        prototypes: Optional[Dict[str, Prototype]] = None,
        config: Optional[PipelineConfig] = None,
        embedder: Optional[TextEmbedder] = None,
    ) -> None:
        self.prototypes = prototypes or load_prototypes(None)
        self.config = config or PipelineConfig()
        self.pipeline = Archetopos(self.prototypes, self.config, embedder=embedder)
        self._seed_cache: List[LatentSeed] = []
        self._max_cache = 24
        self._pending_stress_release = 0.0
        self._last_emergent: List[Dict[str, object]] = []
        self._last_stress_release = 0.0
        self._last_cache_depth = 0
        self._psychoid_sampler = PsychoidArchetypeSampler(self.prototypes)
        self._last_psychoid_signal: Optional[PsychoidSignal] = None

    @staticmethod
    def _payload(question: str, draft: Optional[str]) -> str:
        payload = f"Question: {question.strip()}"
        if draft:
            payload = f"{payload}\nDraft: {draft.strip()}"
        return payload

    def _vectorize(self, payload: str) -> List[float]:
        return list(self.pipeline.embedder.encode([payload])[0])

    def _trim_cache(self) -> None:
        if len(self._seed_cache) <= self._max_cache:
            return
        self._seed_cache.sort(key=lambda seed: seed.created_at)
        self._seed_cache = self._seed_cache[-self._max_cache :]

    def _harvest_emergent(self, vector: List[float]) -> List[EmergentIdea]:
        ideas: List[EmergentIdea] = []
        survivors: List[LatentSeed] = []
        for seed in self._seed_cache:
            similarity = cosine(vector, seed.vector)
            incubation = seed.exposures + 1
            threshold = 0.68 + 0.12 * min(1.0, seed.novelty)
            if similarity >= threshold and incubation >= 2 and seed.intensity >= 0.08:
                ideas.append(
                    EmergentIdea(
                        archetype=seed.archetype_id,
                        label=seed.archetype_label,
                        intensity=round(seed.intensity, 4),
                        incubation_rounds=incubation,
                        trigger_similarity=round(similarity, 4),
                        origin=seed.short_origin(),
                    )
                )
            else:
                seed.exposures = incubation
                seed.intensity *= 0.96
                if seed.intensity >= 0.05:
                    survivors.append(seed)
        self._seed_cache = survivors
        return ideas

    def analyse(
        self,
        *,
        question: str,
        draft: Optional[str] = None,
    ) -> EventMapping:
        """Return the archetypal profile while surfacing emergent insights."""

        payload = self._payload(question, draft)
        mapping = self.pipeline.map_events([payload])[0]
        vector = self._vectorize(payload)
        emergent = [idea.as_dict() for idea in self._harvest_emergent(vector)]
        stress_release = self._pending_stress_release
        self._pending_stress_release = 0.0
        self._last_emergent = emergent
        self._last_stress_release = stress_release
        self._last_cache_depth = len(self._seed_cache)
        self._last_psychoid_signal = self._psychoid_sampler.sample_signal(
            mapping, question=question, draft=draft
        )
        return mapping

    def integrate_outcome(
        self,
        *,
        mapping: Optional[EventMapping],
        question: str,
        draft: str,
        final_answer: str,
        success: bool,
        decision_state: Optional[Dict[str, Any]] = None,
        affect: Optional[Dict[str, float]] = None,
        novelty: Optional[float] = None,
        reward: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update caches after the conscious processing loop completes."""

        decision_state = decision_state or {}
        if mapping is None:
            return {
                "seed_cached": False,
                "stress_delta": 0.0,
                "cache_depth": len(self._seed_cache),
            }

        payload = self._payload(question, draft)
        vector = self._vectorize(payload)
        top_score: Optional[ArchetypeScore] = mapping.archetype_map[0] if mapping.archetype_map else None
        archetype_id = top_score.id if top_score else (mapping.top_k[0] if mapping.top_k else "unknown")
        archetype_label = top_score.label if top_score else archetype_id
        intensity = top_score.intensity if top_score else 0.0
        novelty_value = novelty if novelty is not None else float(decision_state.get("novelty", 0.0))
        left_conf = float(decision_state.get("left_conf_raw", 1.0))
        right_source = decision_state.get("right_source")

        stress_val = 0.0
        if affect:
            stress_val += max(0.0, -float(affect.get("valence", 0.0)))
            stress_val += max(0.0, float(affect.get("risk", 0.0)) - 0.5)
        if not success:
            stress_val += 0.2
        if reward is not None and reward < 0.5:
            stress_val += 0.1

        should_cache = False
        if affect and float(affect.get("valence", 0.0)) < -0.25:
            should_cache = False
        else:
            should_cache = (
                (not success)
                or left_conf < 0.55
                or novelty_value > 0.65
                or right_source == "right_model_fallback"
            ) and intensity >= 0.1

        if should_cache:
            seed = LatentSeed(
                question=question,
                draft=draft or "",
                archetype_id=archetype_id,
                archetype_label=archetype_label,
                intensity=float(intensity),
                novelty=float(novelty_value),
                vector=vector,
                created_at=time.time(),
            )
            self._seed_cache.append(seed)
            self._trim_cache()

        if stress_val:
            self._pending_stress_release += stress_val

        self._last_cache_depth = len(self._seed_cache)
        self._psychoid_sampler.integrate_feedback(success=success, reward=reward)
        return {
            "seed_cached": should_cache,
            "stress_delta": stress_val,
            "cache_depth": len(self._seed_cache),
        }

    def summary(self, mapping: EventMapping, top_k: int = 3) -> Dict[str, object]:
        return {
            "top_k": list(mapping.top_k[:top_k]),
            "geometry": dataclasses.asdict(mapping.geometry),
            "archetype_map": [
                {"id": score.id, "label": score.label, "intensity": score.intensity}
                for score in mapping.archetype_map[:top_k]
            ],
            "emergent_ideas": list(self._last_emergent),
            "stress_released": self._last_stress_release,
            "cache_depth": self._last_cache_depth,
            "psychoid_signal": dataclasses.asdict(self._last_psychoid_signal)
            if self._last_psychoid_signal
            else None,
        }


__all__ = [
    "Archetopos",
    "ArchetypeScore",
    "DEMO_EVENTS",
    "EmergentIdea",
    "EventMapping",
    "Geometry",
    "HashEmbedder",
    "LatentSeed",
    "PsychoidArchetypeSampler",
    "PsychoidSignal",
    "PipelineConfig",
    "Prototype",
    "TextEmbedder",
    "UnconsciousField",
    "build_parser",
    "cmd_demo",
    "cmd_map",
    "cmd_plot",
    "cmd_tune",
    "cosine",
    "export_csv",
    "export_json",
    "load_prototypes",
    "main",
    "plot_polar",
    "save_prototypes",
    "softmax",
]
