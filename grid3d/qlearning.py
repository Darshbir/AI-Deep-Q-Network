# grid3d/qlearning.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

def epsilon_greedy(Q: np.ndarray, s: int, epsilon: float, rng: np.random.RandomState) -> int:
    if rng.rand() < epsilon:
        return rng.randint(Q.shape[1])
    qs = Q[s]
    maxq = qs.max()
    argmax = np.flatnonzero(qs == maxq)
    return int(rng.choice(argmax))

def extract_greedy_policy(Q: np.ndarray) -> np.ndarray:
    return Q.argmax(axis=1)

def train_q_learning(
    env,
    episodes: int = 4000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 3000,
    max_steps_per_ep: int = 200,
    seed: int = 123,
):
    rng = np.random.RandomState(seed)
    nS = env.n_states
    nA = env.n_actions
    Q = np.zeros((nS, nA), dtype=np.float32)
    rewards = np.zeros(episodes, dtype=np.float32)
    epsilons = np.zeros(episodes, dtype=np.float32)
    successes = np.zeros(episodes, dtype=np.float32)   # NEW

    def eps_at(t):
        if t >= epsilon_decay_steps:
            return epsilon_end
        frac = t / float(epsilon_decay_steps)
        return epsilon_start + frac * (epsilon_end - epsilon_start)

    for ep in range(episodes):
        s = env.reset()
        done = False
        G = 0.0
        steps = 0
        epsilon = eps_at(ep)
        epsilons[ep] = epsilon

        while not done and steps < max_steps_per_ep:
            a = epsilon_greedy(Q, s, epsilon, rng)
            s2, r, done, _ = env.step(a)
            td_target = r + (0.0 if done else gamma * Q[s2].max())
            Q[s, a] += alpha * (td_target - Q[s, a])
            G += r
            s = s2
            steps += 1

        rewards[ep] = G
        if env.coord(s) == env.goal:   # reached goal
            successes[ep] = 1.0

    info = {
        "rewards": rewards,
        "epsilons": epsilons,
        "successes": successes,  
        "alpha": np.array([alpha]),
        "gamma": np.array([gamma]),
        "episodes": np.array([episodes]),
    }
    return Q, info
