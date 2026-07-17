from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
Geometrija prostora kombinacija — v4 energija OT na naučenoj metriki

v1 Hebbian D. v2 Perez·circ. v3 RLM+exc → s_K.
v4: optimalni transport ENERGIJE (distribucija) na naučenoj ceni
    C_ij = 1 / (W_ij + δ)   (skupa veza = slaba Hebbian)
    π = Sinkhorn(s_prev, s_K, C)  — masa s_prev (last) → s_K (RLM cilj)
    skor = neto tok π + blagi s_K · L · circ

Dijagnostika odstupanja:
  follow = cos(flow, s_K − s_prev)   (putanja prati naučenu energiju?)
  ako follow nizak → odstupanje (nadogradnja, ne novi šum)

Ban last; next. CSV: loto7_4652_k57.csv, seed=39.
Ime: ig_hebbian_v4_energyot.py
"""

import csv
from itertools import combinations
from math import cos, exp, pi
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
LAMBDA_TEMP = 0.35
K_RLM = 5
EPS_EXC = 0.08
DELTA_C = 1e-3
EPS_OT = 0.08
SINKHORN_ITERS = 150
ZENITH = 20.0
PEREZ_A = 4.0
PEREZ_B = 0.6
PEREZ_C = 1.2
PEREZ_D = 2.5
CIRC_PERIOD = 39
CIRC_KAPPA = 0.25
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4652_k57.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def hebbian_weights(draws: np.ndarray, lam: float = LAMBDA_TEMP) -> np.ndarray:
    W = np.zeros((FRONT_N, FRONT_N), dtype=float)
    for d in draws:
        idx = [int(x) - 1 for x in d.tolist()]
        for a, b in combinations(idx, 2):
            W[a, b] += 1.0
            W[b, a] += 1.0
    for t in range(len(draws) - 1):
        a_idx = [int(x) - 1 for x in draws[t].tolist()]
        b_idx = [int(x) - 1 for x in draws[t + 1].tolist()]
        for a in a_idx:
            for b in b_idx:
                if a == b:
                    continue
                W[a, b] += lam
                W[b, a] += lam
    np.fill_diagonal(W, 0.0)
    return W


def energy_distribution(W: np.ndarray) -> np.ndarray:
    D = W.copy()
    row = D.sum(axis=1, keepdims=True)
    row = np.where(row < 1e-18, 1.0, row)
    return D / row


def learned_cost(W: np.ndarray, delta: float = DELTA_C) -> np.ndarray:
    """Cena: slaba Hebbian veza = skup transport."""
    return 1.0 / (W + delta)


def sinkhorn(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> np.ndarray:
    K = np.exp(-C / EPS_OT)
    u = np.ones(FRONT_N)
    v = np.ones(FRONT_N)
    for _ in range(SINKHORN_ITERS):
        u = a / np.clip(K @ v, 1e-18, None)
        v = b / np.clip(K.T @ u, 1e-18, None)
    return (u[:, None] * K) * v[None, :]


def net_flow(pi: np.ndarray) -> np.ndarray:
    return pi.sum(axis=0) - pi.sum(axis=1)


def perez_luminance(sun: float) -> np.ndarray:
    L = np.zeros(FRONT_N)
    for i in range(FRONT_N):
        n = i + 1
        gamma = abs(n - sun) / float(FRONT_N)
        theta = abs(n - ZENITH) / float(FRONT_N)
        L[i] = (1.0 + PEREZ_C * exp(-PEREZ_A * gamma * gamma)) * (
            1.0 + PEREZ_B * exp(-PEREZ_D * theta * theta)
        )
    return L / L.sum()


def circadian_field(t_index: int) -> np.ndarray:
    phi = 2.0 * pi * (t_index % CIRC_PERIOD) / float(CIRC_PERIOD)
    circ = np.zeros(FRONT_N)
    for i in range(FRONT_N):
        circ[i] = 1.0 + CIRC_KAPPA * cos(phi + 2.0 * pi * i / float(FRONT_N))
    return circ


def excite(s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise = rng.random(FRONT_N) * np.sqrt(np.clip(s, 1e-12, None))
    e = s + noise
    return e / e.sum()


def rlm_walk(D: np.ndarray, last: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    s = np.zeros(FRONT_N)
    for x in last.tolist():
        s[int(x) - 1] = 1.0 / FRONT_SELECT
    for _ in range(K_RLM):
        s = s @ D
        s = np.clip(s, 0.0, None)
        s = s / s.sum() if s.sum() > 0 else np.ones(FRONT_N) / FRONT_N
        s = (1.0 - EPS_EXC) * s + EPS_EXC * excite(s, rng)
        s = s / s.sum()
    return s


def follow_score(flow: np.ndarray, delta_s: np.ndarray) -> float:
    """cos(flow, Δs) — da li OT tok prati RLM pomeraj energije."""
    a = flow - flow.mean()
    b = delta_s - delta_s.mean()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def number_scores(
    flow: np.ndarray,
    s_k: np.ndarray,
    L: np.ndarray,
    circ: np.ndarray,
    ban: set[int],
) -> dict[int, float]:
    out = {}
    for i in range(FRONT_N):
        n = i + 1
        if n in ban:
            out[n] = -1e18
        else:
            out[n] = float(flow[i] + 0.25 * s_k[i] * L[i] * circ[i])
    return out


def _combo_fit(combo, score, target_sum, pos_means, target_odd, ban):
    nums = sorted(combo)
    if any(x in ban for x in nums):
        return -1e18
    s = sum(score[x] for x in nums)
    s -= 0.08 * abs(sum(nums) - target_sum)
    s -= 0.04 * sum(abs(nums[i] - pos_means[i]) for i in range(FRONT_SELECT))
    odd = sum(1 for x in nums if x % 2)
    s -= 0.3 * abs(odd - target_odd)
    return s


def predict_next(draws, score, ban):
    ranked = sorted((n for n in score if n not in ban), key=lambda n: (-score[n], n))
    target_sum = float(draws.sum(axis=1).mean())
    pos_means = [float(draws[:, i].mean()) for i in range(FRONT_SELECT)]
    target_odd = float(np.mean([sum(1 for x in d if x % 2) for d in draws]))
    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, len(ranked) - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))
    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, target_sum, pos_means, target_odd, ban)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, target_sum, pos_means, target_odd, ban)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_v4(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    last = draws[-1]
    ban = set(int(x) for x in last.tolist())
    t_now = len(draws) - 1
    sun = float(np.mean(last))

    W = hebbian_weights(draws)
    D = energy_distribution(W)
    s_prev = np.zeros(FRONT_N)
    for x in last.tolist():
        s_prev[int(x) - 1] = 1.0 / FRONT_SELECT
    s_k = rlm_walk(D, last)
    C = learned_cost(W)
    pi = sinkhorn(s_prev, s_k, C)
    flow = net_flow(pi)
    delta_s = s_k - s_prev
    follow = follow_score(flow, delta_s)
    L = perez_luminance(sun)
    circ = circadian_field(t_now)
    score = number_scores(flow, s_k, L, circ, ban)
    combo = predict_next(draws, score, ban)

    print(f"CSV: {csv_path.name}")
    print(
        f"Kola: {len(draws)} | seed={SEED} | K={K_RLM} | ig_hebbian_v4 energyOT"
    )
    print(f"last: {last.tolist()}")
    print()
    print("=== energija OT + odstupanje ===")
    print(
        {
            "ot_cost": round(float(np.sum(pi * C)), 4),
            "follow_cos": round(follow, 4),
            "follows_nature": follow >= 0.25,
            "flow_l2": round(float(np.linalg.norm(flow)), 6),
        }
    )
    print()
    ranked = sorted(
        ((n, float(score[n])) for n in range(1, FRONT_N + 1) if n not in ban),
        key=lambda t: (-t[1], t[0]),
    )
    print("=== top12 skor (neto tok + s_K·L·circ) ===")
    print([(n, round(sc, 6)) for n, sc in ranked[:12]])
    print()
    print("=== next (ig_hebbian_v4) ===")
    print("next:", combo)


if __name__ == "__main__":
    run_v4()



"""
CSV: loto7_4652_k57.csv
Kola: 4652 | seed=39 | K=5 | ig_hebbian_v4 energyOT
last: [7, 8, 14, 15, 17, 23, 32]

=== energija OT + odstupanje ===
{'ot_cost': 0.0042, 'follow_cos': 1.0, 'follows_nature': True, 'flow_l2': 0.343153}

=== top12 skor (neto tok + s_K·L·circ) ===
[(2, 0.027707), (4, 0.027519), (37, 0.027308), (11, 0.027198), (18, 0.027022), (25, 0.026999), (21, 0.026946), (31, 0.026916), (34, 0.026912), (39, 0.026746), (28, 0.026511), (22, 0.026464)]

=== next (ig_hebbian_v4) ===
next: [4, x, 16, y, 25, z, 37]
"""



"""
OT energije na Hebbian ceni C=1/(W+δ); follow_cos dijagnostika; next.
"""



"""
0. Granica
Loto i.i.d. → nema prediktivnog transporta kao u 03–05.
Ovde: algoritam uči geometriju prostora kombinacija i traži putanju energije (distribucija) → next. Ne LLM.
1. Prostor
Tačka = 7-torica (ili simplex masa na {1…39}).
Manifold = geometrija naučena iz istorije CSV (sličnost / metrika među kolima), ne nametnuti Fisher/Γ.
2. „Nebo“ (Perez intuicija)
Polje „osvetljenosti“ na prostoru brojeva/zona — analog Perez (zenit / sunce / turbidnost → parametri iz podataka).
Cirkadijalni sloj: periodična modulacija polja kroz vreme (indeks kola / faza).
3. Hebbian
Jačanje veza između ko-pojavljivanja / susednih kola na manifoldu („fire together → wire together“).
Matrica / težine = lokalna geometrija.
4. RLM (ne LLM)
Rekurzivno / lokalno učenje putanje na tom grafu/manifoldu (stanje → korak → ažuriranje).
Ekscitacija: mali perturbatori da se održi observabilnost geometrije (Stošić intuicija).
5. Energija = distribucija
Cilj: pomeraj mase/energije (distribucija na simpleksu), ne rang frekvencije.
Putanja ≈ diskretni OT korak na naučenoj metriki (Hebbian+RLM), ne sirovi Sinkhorn kao „predikcija“.
6. next
Kraj putanje / maksimum energije pod zabranom last → jedna kombinacija.
Merilo: gde putanja prati empiriju vs gde odstupa → tada nadogradnja (ne novi šum).

v1 — prostor + Hebbian težine + next
v2 — Perez-polje + cirkadijalna faza
v3 — RLM koraci na manifoldu + ekscitacija
v4 — energija/distribucija OT na naučenoj metriki → next + dijagnostika odstupanja

Seed 39, CSV loto7_4650_k56, samo simulator/.
"""
